"""
Camera streaming routes (WebSocket and HTTP) and capture helpers.
"""

import asyncio
import base64
import io
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

try:
    import matplotlib
except ImportError:
    matplotlib = None

from spark_real.routes import state
from spark_real.routes.visualization import depth_to_colormap

logger = logging.getLogger("spark_server")
router = APIRouter()

# Dedicated pool for stream captures. The frontend polls /api/capture/stream
# every ~66ms and each request can block up to 2s per dead camera; on the
# DEFAULT executor those requests would pile up threadpool workers and starve
# the same pool /api/execute and /api/detect_approve depend on.
from concurrent.futures import ThreadPoolExecutor as _TPE  # noqa: E402

_STREAM_POOL = _TPE(max_workers=3, thread_name_prefix="stream-capture")

# BACKPRESSURE. The executor behind _STREAM_POOL has an UNBOUNDED queue: the
# frontend posts /api/capture/stream every ~66 ms, and one stale camera makes
# capture_*_frame_sync block for seconds (2 s staleness deadline per camera
# plus the inter-camera sleeps), so requests pile up faster than they drain
# and the backlog outlives the stall. Every queued capture is another
# concurrent reader on cameras that are already unhealthy. Admit one stream
# capture at a time and answer the rest with 204 immediately -- the frames
# are identical anyway, so a dropped poll costs nothing.
_STREAM_INFLIGHT = threading.Semaphore(1)

# Liveness: the stream loop is the only thing touching the cameras when the
# rig is idle, which is the state the host has died in.
from spark_real import health_beat as _beat  # noqa: E402

_beat.start()


def _bimanual_registry(pipeline):
    """
    Return the bimanual camera registry if the active pipeline has one.

    The bimanual pipeline attaches `pipeline.camera_registry` at init
    time; single-arm pipelines do not. Callers fall back to the legacy
    three-slot model when this returns None.
    """
    return getattr(pipeline, "camera_registry", None)


def _set_ir(device, enabled: bool) -> None:
    """
    Tell a Kinect whether to build its IR preview image.

    The capture loop computes IR only on request (a per-frame percentile +
    power + stack, consumed only in ir mode). Idempotent and
    cheap; silently a no-op on devices that predate the flag (bimanual
    registry entries, USB cameras, fakes).
    """
    setter = getattr(device, "set_ir_enabled", None)
    if setter is not None:
        try:
            setter(enabled)
        except Exception:  # noqa: BLE001
            logger.debug("set_ir_enabled failed", exc_info=True)


# A stream capture that waited this long for a device lock was contending
# with a detect (or another stream client), which is worth saying out loud.
_STREAM_LOCK_WARN_S = 0.5


def _warn_slow_lock(cam_name, t_requested):
    """Log when a stream capture had to wait for a camera's read lock.

    The streaming path and the detect path share the per-device locks. When
    the two contend, the only external symptom is that everything gets slower,
    with nothing in the log attributing it -- so record the wait at the one
    place that knows it happened.
    """
    waited = time.perf_counter() - t_requested
    if waited > _STREAM_LOCK_WARN_S:
        logger.warning(
            "[detect-timing] stream capture waited %.2fs for %s read lock "
            "(contending with a detect or another stream client)",
            waited,
            cam_name,
        )


def capture_single_camera(
    cam_name: str, need_depth: bool = True, ir_as_rgb: bool = False
):
    """
    Capture from one camera. Returns (rgb, depth) or (None, None).

    If ir_as_rgb is True and the camera is a Kinect, returns the IR
    image in place of the RGB channel (as a 3-channel grayscale frame).
    Used for low-light preview/detection where the colour sensor is
    unusable but the active-IR depth sensor still gives a viewable
    image. The wrist RealSense has no IR substitute via this helper;
    fall back to its normal RGB.

    Bimanual pipelines (with `camera_registry`) dispatch by role
    (`external`, `wrist_left`, `wrist_right`) before falling back to
    the single-arm slot names.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return None, None

    # Bimanual registry path: roles ("external", "wrist_left", "wrist_right").
    reg = _bimanual_registry(pipeline)
    if reg is not None and cam_name in reg.cameras:
        entry = reg.cameras[cam_name]
        try:
            return entry.device.read(depth=need_depth)
        except Exception:
            logger.exception("bimanual camera %r read failed", cam_name)
            return None, None

    # Legacy single-arm slot dispatch. All device reads must serialize
    # through the per-device locks owned by the pipeline. pyk4a/libk4a
    # is not thread-safe across concurrent .read() calls on the same
    # handle, so streaming and detect must not race the same Kinect.
    # Kinect branches wrapped like the RealSense one below: a read exception
    # must degrade to (None, None) (placeholder tile), not repeated 500s on
    # /api/capture/stream.
    _beat.tick("stream")
    if cam_name == "sideview" and pipeline._kinect is not None:
        try:
            _t_lock = time.perf_counter()
            with pipeline._kinect_read_lock:
                _warn_slow_lock("sideview", _t_lock)
                _set_ir(pipeline._kinect, ir_as_rgb)
                if ir_as_rgb:
                    rgb, depth, ir = pipeline._kinect.read(depth=need_depth, ir=True)
                    rgb = ir if ir is not None else rgb
                else:
                    rgb, depth = pipeline._kinect.read(depth=need_depth)
            return rgb, _corrected_depth(depth, pipeline._kinect_cal)
        except Exception:
            logger.exception("sideview kinect read failed")
            return None, None
    elif cam_name == "birdview" and pipeline._kinect2 is not None:
        try:
            _t_lock = time.perf_counter()
            with pipeline._kinect2_read_lock:
                _warn_slow_lock("birdview", _t_lock)
                _set_ir(pipeline._kinect2, ir_as_rgb)
                if ir_as_rgb:
                    rgb, depth, ir = pipeline._kinect2.read(depth=need_depth, ir=True)
                    rgb = ir if ir is not None else rgb
                else:
                    rgb, depth = pipeline._kinect2.read(depth=need_depth)
            return rgb, _corrected_depth(depth, pipeline._kinect2_cal)
        except Exception:
            logger.exception("birdview kinect read failed")
            return None, None
    elif cam_name == "wrist" and pipeline._realsense is not None:
        try:
            with pipeline._realsense_read_lock:
                rgb, depth = pipeline._realsense.read()
            return rgb, _corrected_depth(depth, pipeline._realsense_cal)
        except Exception:
            return None, None
    return None, None


def _corrected_depth(depth, cal):
    """Depth with the per-camera z*scale + offset hand-eye bias applied.

    pipeline.capture() corrects depth before publishing it, and set_last_frame
    makes these frames what click/box detects deproject through
    (routes/detection._snapshot_frame). The hand-eye extrinsic was solved
    WITH the correction in the loop, so skipping it costs real millimeters:
    birdview's depth_scale=0.9879 leaves raw depth ~19 mm long at the 1.55 m
    working range. Single implementation lives on
    CameraCalibration.correct_depth.
    """
    if depth is None or cal is None:
        return depth
    return cal.correct_depth(depth)


def get_camera_order():
    """
    Return the ordered camera role list for the active pipeline.

    Bimanual pipelines surface their registry's `roles` directly
    (typically `["external", "wrist_left", "wrist_right"]`). Single-arm
    pipelines use the historical three slots.
    """
    pipeline = state.pipeline
    if pipeline is not None:
        reg = _bimanual_registry(pipeline)
        if reg is not None and reg.roles:
            return list(reg.roles)
    return ["sideview", "birdview", "wrist"]


def capture_frame_sync():
    """
    Capture cameras, tile side-by-side or show single camera full-frame.

    need_depth IS NOT A BANDWIDTH CONTROL. Depth mode is fixed
    when the device is opened, so both Kinects transmit depth at the
    configured fps no matter what any reader asks for; need_depth=False
    only nulls the depth slot of the returned tuple
    (AzureKinectCamera.read). The knobs that actually change USB traffic
    are kinect_resolution / kinect_fps / kinect_depth_mode, enforced by
    spark_real.usb_budget. need_depth is kept purely to avoid handing
    tiles a depth array they will not draw.

    When state.stream_mode == "ir", Kinect tiles render the active-IR
    image instead of the colour sensor; useful for low-light tests
    since the Kinect projector lights its own scene. Wrist (RealSense)
    has no equivalent and falls back to its normal RGB.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return None

    target_h = 480
    cam_order = get_camera_order()
    ir_mode = state.stream_mode == "ir"

    # Single camera mode (legacy or registry-mediated)
    if state.stream_camera != "all" and state.stream_camera in cam_order:
        cn = state.stream_camera
        rgb, depth = capture_single_camera(cn, need_depth=True, ir_as_rgb=ir_mode)
        if rgb is None:
            return None
        h, w = rgb.shape[:2]
        state.tile_layout = [
            {"cam": cn, "x_start": 0, "width": w, "orig_w": w, "orig_h": h}
        ]
        state.set_last_frame(rgb, depth, cn)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    # Tiled mode: all cameras side-by-side
    results = {}
    for cn in cam_order:
        need_depth = cn == "birdview"
        results[cn] = capture_single_camera(
            cn, need_depth=need_depth, ir_as_rgb=ir_mode
        )

    tiles = []
    layout = []
    x_offset = 0
    for cn in cam_order:
        rgb, depth = results.get(cn, (None, None))
        if rgb is not None:
            h, w = rgb.shape[:2]
            new_w = int(w * target_h / h)
            tile = np.array(
                Image.fromarray(rgb).resize((new_w, target_h), Image.BILINEAR)
            )
            tiles.append(tile)
            layout.append(
                {
                    "cam": cn,
                    "x_start": x_offset,
                    "width": new_w,
                    "orig_w": w,
                    "orig_h": h,
                }
            )
            x_offset += new_w
            if cn == state.active_camera:
                state.set_last_frame(rgb, depth, cn)
        else:
            placeholder = np.zeros((target_h, int(target_h * 4 / 3), 3), dtype=np.uint8)
            placeholder[target_h // 2 - 10 : target_h // 2 + 10, :] = 30
            pw = placeholder.shape[1]
            tiles.append(placeholder)
            layout.append(
                {
                    "cam": cn,
                    "x_start": x_offset,
                    "width": pw,
                    "orig_w": pw,
                    "orig_h": target_h,
                }
            )
            x_offset += pw
    state.tile_layout = layout

    if not any(t.any() for t in tiles):
        return None

    if state.get_last_frame()[0] is None:
        for cn in cam_order:
            rgb, depth = results.get(cn, (None, None))
            if rgb is not None:
                state.set_last_frame(rgb, depth, cn)
                break

    combined = np.hstack(tiles)
    img = Image.fromarray(combined)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def capture_depth_frame_sync():
    """
    Capture depth from all cameras, apply turbo colormap, tile.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return None

    target_h = 480
    cam_order = get_camera_order()
    # matplotlib import is allowed to fail at module import; don't
    # dereference None here (it took the whole depth stream down).
    if matplotlib is None:
        return None
    cmap = matplotlib.colormaps["turbo"]

    results = {}
    for cn in cam_order:
        results[cn] = capture_single_camera(cn, need_depth=True)
        if cn != cam_order[-1]:
            time.sleep(0.05)

    tiles = []
    for cn in cam_order:
        rgb, depth = results.get(cn, (None, None))
        if depth is not None and depth.max() > 0:
            d = depth.copy().astype(np.float32)
            valid = d > 0.01
            if valid.any():
                vmin, vmax = np.percentile(d[valid], [2, 98])
                d = np.clip((d - vmin) / max(vmax - vmin, 0.01), 0, 1)
                d[~valid] = 0
            else:
                d[:] = 0
            colored = (cmap(d)[:, :, :3] * 255).astype(np.uint8)
            h, w = colored.shape[:2]
            new_w = int(w * target_h / h)
            tile = np.array(
                Image.fromarray(colored).resize((new_w, target_h), Image.BILINEAR)
            )
            tiles.append(tile)
        elif rgb is not None:
            h, w = rgb.shape[:2]
            new_w = int(w * target_h / h)
            tile = np.zeros((target_h, new_w, 3), dtype=np.uint8)
            tile[:, :, 2] = 30
            tiles.append(tile)
        else:
            tiles.append(np.zeros((target_h, int(target_h * 4 / 3), 3), dtype=np.uint8))

    if not tiles:
        return None

    combined = np.hstack(tiles)
    img = Image.fromarray(combined)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def get_available_cameras() -> list:
    """
    Return list of available camera role names.

    Bimanual pipelines: pull the role list directly from the camera
    registry (e.g. ['external', 'wrist_left', 'wrist_right']). Single-arm:
    enumerate the three named slots as before. The 'depth' pseudo-camera
    is appended only when at least one real camera is present so the
    frontend's depth-stream toggle has something to render.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return []
    reg = _bimanual_registry(pipeline)
    if reg is not None and reg.roles:
        return list(reg.roles) + ["depth"]
    cams = []
    if pipeline._kinect is not None:
        cams.append("sideview")
    if pipeline._kinect2 is not None:
        cams.append("birdview")
    if pipeline._realsense is not None:
        cams.append("wrist")
    if cams:
        cams.append("depth")
    return cams


@router.get("/api/capture/stream")
async def capture_stream():
    """
    Fast endpoint: returns raw JPEG of tiled cameras (RGB or depth).

    Drops the poll (204) when a capture is already in flight; see
    _STREAM_INFLIGHT.
    """
    if not _STREAM_INFLIGHT.acquire(blocking=False):
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    try:
        loop = asyncio.get_event_loop()
        if state.stream_mode == "depth":
            jpeg = await loop.run_in_executor(_STREAM_POOL, capture_depth_frame_sync)
        else:
            jpeg = await loop.run_in_executor(_STREAM_POOL, capture_frame_sync)
    finally:
        _STREAM_INFLIGHT.release()
    if jpeg is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/api/capture/stereo")
async def capture_stereo():
    """
    ZED stereo pair (left + right PNG, base64) plus SDK intrinsics.

    Used by calibration.bimanual_anchor_tap to grab stereo frames from the
    running server without opening a second ZED handle (the SDK is
    exclusive). Only the bimanual registry's "external" camera exposes
    read_stereo; single-arm pipelines answer 400.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "pipeline not ready"})
    reg = _bimanual_registry(pipeline)
    entry = reg.get("external") if reg is not None else None
    zed_cam = getattr(entry, "device", entry)
    if zed_cam is None or not hasattr(zed_cam, "read_stereo"):
        return JSONResponse(status_code=400, content={"error": "no ZED stereo"})

    def _grab():
        left, right, _depth = zed_cam.read_stereo()
        _, buf_l = cv2.imencode(".png", left)
        _, buf_r = cv2.imencode(".png", right)
        return {
            "left_png_b64": base64.b64encode(buf_l.tobytes()).decode(),
            "right_png_b64": base64.b64encode(buf_r.tobytes()).decode(),
            "info": zed_cam.get_camera_info(),
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_STREAM_POOL, _grab)
    return JSONResponse(content=result)


@router.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):
    """
    Stream camera frames as binary JPEG over WebSocket at ~15-20 FPS.
    """
    await websocket.accept()
    pipeline = state.pipeline
    if pipeline is None:
        await websocket.send_json({"error": "Pipeline not initialized"})
        await websocket.close()
        return
    loop = asyncio.get_event_loop()
    try:
        while True:
            t0 = time.time()
            jpeg_bytes = await loop.run_in_executor(_STREAM_POOL, capture_frame_sync)
            if jpeg_bytes:
                await websocket.send_bytes(jpeg_bytes)
            elapsed = time.time() - t0
            # 15 fps cap: the Kinects produce 15 fps, so encoding faster than
            # that just re-encodes identical cached frames.
            await asyncio.sleep(max(0.01, 0.0667 - elapsed))
    except WebSocketDisconnect:
        pass
    except Exception:
        # A silently dying stream is indistinguishable from a crash on the
        # frontend; leave a traceback so the next 'camera froze' report has
        # log evidence.
        logger.exception("/ws/camera stream loop died")
