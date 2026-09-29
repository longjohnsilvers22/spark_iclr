"""V-JEPA 2-AC loader and forward-pass wrappers.

Frozen pretrained checkpoint from Meta (Apache-2 / MIT mix; encoder code MIT,
weights CC-BY-NC 4.0 per upstream LICENSE). Used here for zero-training
BT verification: encode current + goal RGB frames, roll candidate trajectories
through the action-conditioned predictor, score L2 distance to goal latent.

Upstream code lifted with provenance comments. Pinned commit:
    facebookresearch/vjepa2 @ 204698b45b3712590f06245fbfba32d3be539812

Provenance:
- load_vjepa2_ac(): mirrors notebooks/energy_landscape_example.ipynb cells 2-3
- encode_frame(): adapted from notebooks/utils/world_model_wrapper.py:42-52
                   (WorldModel.encode) and forward_target() in the same notebook.
- predict_next_latent(): mirrors step_predictor() in
                          notebooks/utils/world_model_wrapper.py:56-64.

References:
- Assran et al. 2025, "V-JEPA 2: Self-Supervised Video Models Enable
  Understanding, Prediction and Planning", arXiv:2506.09985.
- Upstream repo: https://github.com/facebookresearch/vjepa2
- AC checkpoint URL: https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_REPO = Path(__file__).resolve().parents[3]
VJEPA2_REPO_ROOT = Path(os.environ.get("VJEPA2_REPO_ROOT", _REPO / "external" / "vjepa2"))
VJEPA2_CKPT_DIR = Path(os.environ.get("VJEPA2_CKPT_DIR", _REPO / "external" / "vjepa2_ckpts"))
# Upstream sets VJEPA_BASE_URL = "http://localhost:8300" for testing in
# src/hub/backbones.py:11.  Overridden here to the real public CDN.
VJEPA_BASE_URL_REAL = "https://dl.fbaipublicfiles.com/vjepa2"


def _patch_upstream_url() -> None:
    """Patch the upstream backbone module so it pulls from the real CDN."""
    if str(VJEPA2_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(VJEPA2_REPO_ROOT))
    # Pre-import the module and overwrite its module-level URL.
    from src.hub import backbones as _bb  # noqa: WPS433  (local import for patch)

    _bb.VJEPA_BASE_URL = VJEPA_BASE_URL_REAL


@dataclass
class VJEPA2AC:
    """Bundle of encoder, action-conditioned predictor, and image transform."""

    encoder: torch.nn.Module
    predictor: torch.nn.Module
    transform: object  # callable: TxHxWxC uint8/float -> CxTxHxW normalized
    crop_size: int = 256
    patch_size: int = 16
    tokens_per_frame: int = 256  # (crop_size // patch_size) ** 2
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


def load_vjepa2_ac(
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_local_ckpt: bool = True,
) -> VJEPA2AC:
    """Load encoder + AC predictor and the canonical 256x256 transform.

    Upstream: notebooks/energy_landscape_example.ipynb cells 2-3.

    Args:
        device: cuda device string
        dtype: bf16 on RTX 5090 fits ViT-g/16 (~7 GB)
        use_local_ckpt: if True and local file exists, monkeypatch torch.hub
                         to read from disk rather than download.

    Returns:
        VJEPA2AC bundle with encoder.eval(), predictor.eval(), transform.
    """
    _patch_upstream_url()

    # Optional: pre-stash the checkpoint in torch hub cache so the factory
    # picks it up via load_state_dict_from_url without re-downloading.
    local_ckpt = VJEPA2_CKPT_DIR / "vjepa2-ac-vitg.pt"
    if use_local_ckpt and local_ckpt.exists():
        # torch.hub uses TORCH_HOME / hub / checkpoints / <basename>
        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        hub_dir.mkdir(parents=True, exist_ok=True)
        # The upstream factory queries URL <BASE>/vjepa2-ac-vitg.pt; the
        # cache filename torch.hub uses is the URL basename (Python urllib).
        cached = hub_dir / "vjepa2-ac-vitg.pt"
        if not cached.exists() or cached.stat().st_size != local_ckpt.stat().st_size:
            try:
                if cached.exists():
                    cached.unlink()
                cached.symlink_to(local_ckpt.resolve())
            except OSError:
                # Fall back to copy if symlink fails (e.g. cross-fs).
                import shutil

                shutil.copy2(local_ckpt, cached)

    # Upstream: hubconf.py exposes vjepa2_ac_vit_giant -> src/hub/backbones.py:
    # vjepa2_ac_vit_giant() -> _make_vjepa2_ac_model(model_name="vit_ac_giant").
    from src.hub.backbones import vjepa2_ac_vit_giant  # type: ignore

    encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()

    # Upstream: notebooks/energy_landscape_example.ipynb cell 2 ->
    #          app/vjepa_droid/transforms.py:make_transforms with
    #          crop_size=256, no augment.
    from app.vjepa_droid.transforms import make_transforms  # type: ignore

    crop_size = 256
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    patch_size = encoder.patch_size if hasattr(encoder, "patch_size") else 16
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    return VJEPA2AC(
        encoder=encoder,
        predictor=predictor,
        transform=transform,
        crop_size=crop_size,
        patch_size=patch_size,
        tokens_per_frame=tokens_per_frame,
        device=device,
        dtype=dtype,
    )


def encode_frame(
    wm: VJEPA2AC,
    image: np.ndarray,
    *,
    normalize_reps: bool = True,
) -> torch.Tensor:
    """Encode a single HxWx3 uint8 RGB image to ViT-g/16 patch features.

    Upstream: notebooks/utils/world_model_wrapper.py:42-52  WorldModel.encode

    The upstream pipeline expects video clips; for image goals we repeat
    the single frame 2x along the temporal axis (matches the tubelet_size=2
    inflate the encoder applies internally) -- this is the same trick the
    HF model card uses to encode a still image:
        pixel_values = pixel_values.repeat(1, 16, 1, 1, 1)

    Args:
        wm: loaded bundle
        image: HxWx3 uint8 RGB (any size; transform crops to 256x256)
        normalize_reps: apply LayerNorm to outputs (matches notebook default)

    Returns:
        Tensor of shape [1, tokens_per_frame, embed_dim] on wm.device.
    """
    # transform expects a TxHxWxC numpy array (single clip)
    clip = np.expand_dims(image, axis=0)  # [1, H, W, 3]
    clip_t = wm.transform(clip)[None, :]  # [1, C, T=1, H, W]
    B, C, T, H, W = clip_t.size()

    # Upstream world_model_wrapper.py:46 -- reshape and repeat tubelet
    # dimension so tubelet_size=2 conv finds 2 frames.
    clip_t = (
        clip_t.permute(0, 2, 1, 3, 4)  # B T C H W
        .flatten(0, 1)  # (B*T) C H W
        .unsqueeze(2)  # (B*T) C 1 H W
        .repeat(1, 1, 2, 1, 1)  # (B*T) C 2 H W
    )
    clip_t = clip_t.to(device=wm.device, dtype=wm.dtype, non_blocking=True)

    with torch.no_grad():
        h = wm.encoder(clip_t)  # [(B*T), N_per_frame, D]
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # [B, T*N_per_frame, D]
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def predict_next_latent(
    wm: VJEPA2AC,
    z_ctx: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    *,
    normalize_reps: bool = True,
) -> torch.Tensor:
    """One predictor step: (z_ctx, action, state) -> z_next.

    Upstream: notebooks/utils/world_model_wrapper.py:56-64  step_predictor()
              and notebooks/energy_landscape_example.ipynb step_predictor().

    The predictor reads a flat (T * tokens_per_frame, D) context tensor.
    Output is the last frame's tokens only (tokens_per_frame, D).

    Args:
        z_ctx: [B, T*tokens_per_frame, D] context patch features
        actions: [B, T, 7] per-frame 7-D Cartesian + gripper delta
        states: [B, T, 7] per-frame 7-D EE pose (xyz, euler, gripper)

    Returns:
        z_next: [B, tokens_per_frame, D] predicted next-frame patch features
    """
    actions = actions.to(device=wm.device, dtype=wm.dtype, non_blocking=True)
    states = states.to(device=wm.device, dtype=wm.dtype, non_blocking=True)
    z_ctx = z_ctx.to(device=wm.device, dtype=wm.dtype, non_blocking=True)

    with torch.no_grad():
        # src/models/ac_predictor.py:136 forward(x, actions, states, extrinsics)
        z_pred_all = wm.predictor(z_ctx, actions, states)
    z_next = z_pred_all[:, -wm.tokens_per_frame :]
    if normalize_reps:
        z_next = F.layer_norm(z_next, (z_next.size(-1),))
    return z_next


def unroll_actions(
    wm: VJEPA2AC,
    z0: torch.Tensor,
    s0: torch.Tensor,
    action_seq: torch.Tensor,
    *,
    normalize_reps: bool = True,
    max_rollout: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll an action sequence forward from (z0, s0) through the predictor.

    Upstream: notebooks/energy_landscape_example.ipynb forward_actions()
              (the recurrent loop), simplified to single-batch unroll.

    DROID training used 16-frame clips at 4 fps (config:
    configs/train/vitg16/droid-256px-8f.yaml -> 8 frames per sample).
    To stay within the training distribution the rollout is capped at
    ``max_rollout`` predictor steps; if the BT compiles to more 7-DoF
    deltas than that, consecutive deltas are summed into ``max_rollout``
    chunks (preserves the net displacement, halves cadence).

    Args:
        z0: [1, tokens_per_frame, D] initial latent
        s0: [1, 1, 7] initial 7-D EE pose
        action_seq: [T_act, 7] sequence of 7-D EE deltas
        max_rollout: hard cap on predictor unroll depth (DROID config: 8)

    Returns:
        (z_final, s_final): final latent [1, tokens_per_frame, D] and pose [1, 1, 7]
    """
    # Lifted from notebooks/utils/mpc_utils.py:166 compute_new_pose
    from notebooks.utils.mpc_utils import compute_new_pose  # type: ignore

    # -- Chunk action_seq down to <= max_rollout steps by summing deltas.
    T_act = action_seq.shape[0]
    if T_act > max_rollout:
        n_chunks = max_rollout
        chunk_size = int(np.ceil(T_act / n_chunks))
        # Pad to a multiple of chunk_size, then sum within each chunk.
        pad_n = chunk_size * n_chunks - T_act
        if pad_n > 0:
            pad = torch.zeros((pad_n, action_seq.shape[1]),
                              device=action_seq.device, dtype=action_seq.dtype)
            action_seq = torch.cat([action_seq, pad], dim=0)
        chunked = action_seq.view(n_chunks, chunk_size, -1).sum(dim=1)
        action_seq = chunked
        T_act = n_chunks

    z_hat = z0
    s_hat = s0
    a_buf: Optional[torch.Tensor] = None

    for t in range(T_act):
        a_t = action_seq[t : t + 1].unsqueeze(0).to(
            device=wm.device, dtype=wm.dtype
        )  # [1, 1, 7]
        a_buf = a_t if a_buf is None else torch.cat([a_buf, a_t], dim=1)
        # The predictor is block-causal and ingests the full context history.
        # Only the next-frame token is needed, so feed z_hat (which already
        # accumulated history) and the action history.
        z_next = predict_next_latent(
            wm, z_hat, a_buf, s_hat, normalize_reps=normalize_reps
        )
        z_hat = torch.cat([z_hat, z_next], dim=1)
        s_next = compute_new_pose(s_hat[:, -1:], a_t)
        s_hat = torch.cat([s_hat, s_next], dim=1)

    z_final = z_hat[:, -wm.tokens_per_frame :]
    s_final = s_hat[:, -1:]
    return z_final, s_final


# ---------------------------------------------------------------------------
# Convenience scoring
# ---------------------------------------------------------------------------

def latent_l2(z_pred: torch.Tensor, z_goal: torch.Tensor) -> float:
    """Mean per-token L2 distance between two latent maps.

    Both inputs are [1, tokens_per_frame, D].  Returns a scalar Python float.
    """
    diff = (z_pred.float() - z_goal.float()).flatten(1)
    return float(diff.norm(dim=-1).mean().item())


def latent_l1(z_pred: torch.Tensor, z_goal: torch.Tensor) -> float:
    """Mean elementwise L1 distance -- the loss V-JEPA 2-AC was trained with.

    Matches notebooks/utils/mpc_utils.py:17  l1() = mean(|a - b|).
    """
    diff = (z_pred.float() - z_goal.float()).abs()
    return float(diff.mean().item())
