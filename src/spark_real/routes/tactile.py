"""
HTTP routes for FlexiTac tactile sensors.

Exposes the latest tactile frame from each connected sensor as JSON
(for programmatic clients) and as a colour-mapped PNG heatmap (for the
operator UI). All routes 200 with ``connected: false`` when no sensor
is attached; never 404 or 500 just because tactile is disabled.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

try:
    import matplotlib
except ImportError:
    matplotlib = None

from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter()


# Fallback manager for pipeline-less runs (--no-init): tactile viewing
# shouldn't require the full perception/robot stack to be up.
_standalone_mgr = None


def _get_manager():
    pipeline = state.pipeline
    if pipeline is not None:
        return getattr(pipeline, "_tactile", None)
    return _standalone_mgr


@router.post("/api/tactile/reconnect")
async def tactile_reconnect():
    """
    Re-probe serial ports for FlexiTac sensors (e.g. after a replug)
    without restarting the server. TactileManager.start() is idempotent;
    if sensors are already running this is a no-op. Works with or without
    a pipeline: in --no-init runs the manager lives module-level here.
    """
    global _standalone_mgr
    mgr = _get_manager()
    if mgr is None:
        try:
            from spark_real.sensors.tactile import TactileManager
            mgr = TactileManager()
        except Exception as exc:
            return JSONResponse(status_code=200,
                                content={"connected": False, "error": str(exc)})
        if state.pipeline is not None:
            state.pipeline._tactile = mgr
        else:
            _standalone_mgr = mgr
    try:
        mgr.start()
    except Exception as exc:
        return JSONResponse(status_code=200,
                            content={"connected": False, "error": str(exc)})
    return {"connected": mgr.available(), "sides": mgr.sides()}


@router.post("/api/tactile/rebaseline")
async def tactile_rebaseline(side: str | None = None):
    """
    Re-zero: recapture per-cell baselines on connected sensors (~1 s).
    The pads must be UNTOUCHED while this runs - contact during capture
    gets baked into the new zero. Use after replug/re-tape/remount or
    when resting deltas have drifted.
    """
    mgr = _get_manager()
    if mgr is None or not mgr.available():
        return JSONResponse(
            status_code=200,
            content={"connected": False, "error": "no sensors connected"},
        )
    results = await asyncio.to_thread(mgr.rebaseline, side)
    return {
        "connected": True,
        "sides": [{"side": sd, "ok": bool(ok)} for sd, ok in results.items()],
    }


@router.get("/api/tactile/status")
async def tactile_status():
    """
    Quick overview: which sides are connected, current contact counts.
    """
    mgr = _get_manager()
    if mgr is None or not mgr.available():
        return {"connected": False, "sides": []}
    out = {"connected": True, "sides": []}
    for side in mgr.sides():
        s = mgr.snapshot(side)
        out["sides"].append(
            {
                "side": side,
                "rows": s.rows,
                "cols": s.cols,
                "seq": s.seq,
                "timestamp_s": s.timestamp_s,
                "contact_cell_count": s.contact_cell_count,
                "max_response": round(s.max_response, 3),
                "centroid_uv": list(s.centroid_uv) if s.centroid_uv else None,
                "in_contact": mgr.is_in_contact(side),
            }
        )
    return out


@router.get("/api/tactile/frame")
async def tactile_frame(side: str = "left"):
    """
    Latest frame as JSON. Heavy-ish (rows*cols floats), but trivially
    small for FlexiTac V2 (12*32 = 384 cells).
    """
    mgr = _get_manager()
    if mgr is None or not mgr.available(side):
        return JSONResponse(status_code=200, content={"connected": False, "side": side})
    s = mgr.snapshot(side)
    return {
        "connected": True,
        "side": s.side,
        "rows": s.rows,
        "cols": s.cols,
        "seq": s.seq,
        "timestamp_s": s.timestamp_s,
        "normalized": s.normalized.tolist(),
        "contact_cell_count": s.contact_cell_count,
        "max_response": round(s.max_response, 3),
        "centroid_uv": list(s.centroid_uv) if s.centroid_uv else None,
    }


@router.get("/api/tactile/heatmap.png")
async def tactile_heatmap(side: str = "left", scale: int = 16):
    """
    Latest frame as a colour-mapped PNG, suitable for ``<img>`` tag.

    ``scale`` integer-resizes the (rows*cols) grid for a viewable
    overlay; default 16 makes a 12x32 frame into 192x512 px.
    """
    mgr = _get_manager()
    if mgr is None or not mgr.available(side):
        # 1x1 transparent PNG so the UI element doesn't break.
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    s = mgr.snapshot(side)
    # Colour by ABSOLUTE force (raw - baseline) in sensor counts on a fixed
    # full-scale, so brightness reflects real force, not the self-scaled
    # `normalized` (which always peaks at 1.0 on any contact). Empirically
    # ~50 counts = contact threshold, ~150 = firm press.
    try:
        if matplotlib is None:
            raise ImportError("matplotlib not available")
        cmap = matplotlib.colormaps["inferno"]
    except Exception:
        cmap = None
    v = np.clip(s.delta.astype(np.float32) / 150.0, 0.0, 1.0)
    if cmap is not None:
        rgba = (cmap(v) * 255).astype(np.uint8)
    else:
        # Fallback: blue to yellow
        g = (v * 255).astype(np.uint8)
        rgba = np.stack([g, g, 255 - g, np.full_like(g, 255)], axis=-1)
    # Rotate 90 deg counter-clockwise to match the operator's view of the pad.
    rgba = np.rot90(rgba, k=1)
    h, w = rgba.shape[0], rgba.shape[1]
    img = Image.fromarray(rgba, mode="RGBA")
    img = img.resize((w * int(scale), h * int(scale)), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@router.websocket("/ws/tactile")
async def tactile_stream(websocket: WebSocket):
    """
    Push tactile frames for ALL connected sides as JSON at ~30 Hz.

    Message shape: ``{"connected": bool, "sides": [{side, rows, cols, seq,
    timestamp_s, delta, contact_cell_count, max_response, centroid_uv,
    in_contact}]}`` where ``delta`` is the (rows, cols) raw-minus-baseline
    signal in integer sensor counts.

    The socket stays open when no sensor is attached (``connected: false``
    at 1 Hz) so the viewer recovers after /api/tactile/reconnect without a
    page reload.
    """
    await websocket.accept()
    try:
        while True:
            mgr = _get_manager()
            if mgr is None or not mgr.available():
                await websocket.send_json({"connected": False, "sides": []})
                await asyncio.sleep(1.0)
                continue
            sides = []
            for side in sorted(mgr.sides()):
                s = mgr.snapshot(side)
                sides.append(
                    {
                        "side": s.side,
                        "rows": s.rows,
                        "cols": s.cols,
                        "seq": s.seq,
                        "timestamp_s": s.timestamp_s,
                        "delta": np.rint(s.delta).astype(int).tolist(),
                        "contact_cell_count": s.contact_cell_count,
                        "max_response": round(float(s.max_response), 1),
                        "centroid_uv": list(s.centroid_uv) if s.centroid_uv else None,
                        "in_contact": bool(mgr.is_in_contact(side)),
                    }
                )
            await websocket.send_json({"connected": True, "sides": sides})
            await asyncio.sleep(1.0 / 30.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("tactile websocket closed", exc_info=True)


_VIEWER_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "frontend" / "tactile_viewer.html"
)


@router.get("/tactile")
async def tactile_viewer():
    """Standalone live force-map viewer. Open in its own browser window.

    Read fresh from disk on every request so layout/CSS tweaks show up on a
    browser refresh with no server reload.
    """
    try:
        return HTMLResponse(content=_VIEWER_HTML_PATH.read_text())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>tactile_viewer.html not found</h1>", status_code=500
        )
