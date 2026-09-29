"""
MolmoAct pointing worker -- runs INSIDE the ``molmoact2`` conda env.

Loaded checkpoint: ``allenai/MolmoAct2-LIBERO`` (already in the local HF
cache; built on Molmo2-ER, which is the pointing-trained base).  This
worker only uses the VLM's text head (``generate``), never the action
expert, so bf16 keeps it well under the 4090's memory.

Protocol (chosen so the parent never imports torch/transformers):

* one-shot:  ``python molmo_worker.py --image f.png --query "red mug"``
  prints one JSON line ``{"ok": true, "text": "<raw model reply>"}``.
* serve:     ``python molmo_worker.py --serve`` reads one JSON request
  per line on stdin (``{"image": "/path.png", "query": "...",
  "kind": "point"}``) and answers one JSON line per request -- the model
  loads once and stays resident across queries (a per-query reload of a
  ~16 GB checkpoint would make the rescue rung unusable).

The RAW reply text is returned; point-tag parsing stays in the parent
(spark_real.perception.providers.molmo) where it is unit-tested.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

MODEL_ID = "allenai/MolmoAct2-LIBERO"

_MODEL = None
_PROCESSOR = None


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _MODEL = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )


def _prompt_for(query: str, kind: str) -> str:
    if kind == "trace":
        # MolmoAct's visual-reasoning-trace phrasing; the parent parses
        # whatever point list comes back and fails open on prose.
        return (
            f"Show the visual trace of the gripper to accomplish: {query}. "
            "Answer with a list of image points."
        )
    return f"Point to the {query}."


def _answer(image_path: str, query: str, kind: str) -> dict:
    import torch
    from PIL import Image

    _load()
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _prompt_for(query, kind)},
            ],
        }
    ]
    inputs = _PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_MODEL.device)
    with torch.inference_mode():
        out = _MODEL.generate(**inputs, max_new_tokens=256, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    text = _PROCESSOR.batch_decode(
        out[:, n_in:], skip_special_tokens=True
    )[0]
    return {"ok": True, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--query")
    ap.add_argument("--kind", default="point")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    if args.serve:
        # JSONL request/response loop; a bad request answers an error line
        # and keeps serving. EOF on stdin ends the worker.
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                res = _answer(req["image"], req["query"],
                              req.get("kind", "point"))
            except Exception:  # noqa: BLE001 - report, don't die
                res = {"ok": False, "error": traceback.format_exc(limit=3)}
            print(json.dumps(res), flush=True)
        return

    try:
        res = _answer(args.image, args.query, args.kind)
    except Exception:  # noqa: BLE001
        res = {"ok": False, "error": traceback.format_exc(limit=3)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
