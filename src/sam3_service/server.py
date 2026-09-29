"""
SAM3 FastAPI service for SPARK.

Mirrors ``cap_gym/capx/serving/launch_sam3_server.py`` but lives at
``127.0.0.1:8115`` so we don't clash with CaP-X's 8114 service. Loads the
model once, serialises GPU access with an asyncio semaphore, and exposes
both text-prompt and point-prompt endpoints.

Run with:
    bash scripts/launch_sam3_service.sh
or directly:
    python -m sam3_service.server --port 8115
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# Logging + global state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SPARK SAM3 service")

_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"

# Serialise GPU access so concurrent requests don't OOM the 5090.
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """
    Run a blocking GPU function without blocking the event loop.
    """
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# Helpers

def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


def encode_mask_bool(mask: np.ndarray) -> str:
    """
    Pack a boolean mask to bytes (uint8) then base64.
    """
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


# Request / response models

class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]  # [x, y]


class PointPromptResponse(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


# Core inference

def _do_segment(pil_image: Image.Image, text_prompt: str) -> SegmentResponse:
    device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16):
        state = _PROCESSOR.set_image(pil_image)
        out = _PROCESSOR.set_text_prompt(state=state, prompt=text_prompt)

    masks_t = out.get("masks")
    boxes_t = out.get("boxes")
    scores_t = out.get("scores")
    if masks_t is None or boxes_t is None or scores_t is None:
        return SegmentResponse(results=[])

    masks_np = _to_numpy(masks_t)
    boxes_np = _to_numpy(boxes_t)
    scores_np = _to_numpy(scores_t)

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    results: list[MaskData] = []
    for i in range(len(scores_np)):
        mask = masks_np[i] > 0
        if not mask.any():
            continue
        results.append(
            MaskData(
                mask_base64=encode_mask_bool(mask),
                shape=list(mask.shape),
                box=boxes_np[i].tolist(),
                score=float(scores_np[i]),
                label=text_prompt,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return SegmentResponse(results=results)


def _do_segment_point(pil_image: Image.Image,
                       point_xy: tuple[float, float]) -> PointPromptResponse:
    """
    Point-prompt path -- mirrors Sam3Processor.set_point_prompt's
    best-IoU-pick logic so callers get a single mask back.
    """
    device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16):
        state = _PROCESSOR.set_image(pil_image)
        state = _PROCESSOR.set_point_prompt(
            state=state, point_xy=(float(point_xy[0]), float(point_xy[1])),
            label=1, multimask_output=True,
        )

    masks_t = state.get("masks")
    boxes_t = state.get("boxes")
    scores_t = state.get("scores")
    if masks_t is None or scores_t is None or masks_t.numel() == 0:
        return PointPromptResponse(mask_base64="", shape=[0, 0],
                                    box=[0.0, 0.0, 0.0, 0.0], score=0.0)

    mask = _to_numpy(masks_t)[0].astype(bool)
    score = float(_to_numpy(scores_t)[0]) if scores_t.numel() else 0.0
    if boxes_t is not None and boxes_t.numel() >= 4:
        box = _to_numpy(boxes_t)[0].tolist()
    elif mask.any():
        ys, xs = np.where(mask)
        box = [float(xs.min()), float(ys.min()),
               float(xs.max()), float(ys.max())]
    else:
        box = [0.0, 0.0, 0.0, 0.0]

    return PointPromptResponse(
        mask_base64=encode_mask_bool(mask),
        shape=list(mask.shape),
        box=box,
        score=score,
    )


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=_DEVICE,
        model_loaded=(_PROCESSOR is not None and _MODEL is not None),
    )


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest) -> SegmentResponse:
    if _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialised")
    pil_image = decode_image(req.image_base64)
    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt)
    except Exception as exc:
        logger.error("Text segment failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@app.post("/point_prompt", response_model=PointPromptResponse)
async def point_prompt(req: PointPromptRequest) -> PointPromptResponse:
    if _PROCESSOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialised")
    if getattr(_MODEL, "inst_interactive_predictor", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Instance interactivity not enabled on SAM3 model",
        )
    if not isinstance(req.point_coords, (list, tuple)) or len(req.point_coords) != 2:
        raise HTTPException(status_code=400,
                            detail="point_coords must be a 2-element [x, y] list")
    pil_image = decode_image(req.image_base64)
    try:
        return await _run_on_gpu(
            _do_segment_point, pil_image,
            (float(req.point_coords[0]), float(req.point_coords[1])),
        )
    except Exception as exc:
        logger.error("Point prompt failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# Entry point

def main(
    device: str = "cuda",
    port: int = 8115,
    host: str = "127.0.0.1",
    confidence_threshold: float = 0.0,
) -> None:
    """
    Start the SAM3 service. Default port 8115 (CaP-X is on 8114).
    """
    global _MODEL, _PROCESSOR, _DEVICE

    _DEVICE = device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if "cuda" in device:
            idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(idx)

    logger.info("Building SAM3 image model (inst_interactivity=True)...")
    _MODEL = build_sam3_image_model(enable_inst_interactivity=True)
    if hasattr(_MODEL, "to") and device:
        _MODEL = _MODEL.to(device)
    _PROCESSOR = Sam3Processor(_MODEL, device=device,
                                confidence_threshold=confidence_threshold)
    logger.info("SAM3 ready on %s. Listening on http://%s:%d", device, host, port)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
