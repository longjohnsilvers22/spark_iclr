"""
SPARK Real Pipeline: FastAPI Web Server.

Provides REST + WebSocket API for the interactive frontend:
- Camera capture, switching, and live streaming
- SAM3 object detection (text, click/point, box prompts)
- Gemini plan generation
- UR10e execution control
- Pipeline status monitoring

Usage:
    cd ~/spark/src
    python -m spark_real.server [--port 8888] [--no-robot]
"""

import logging
import os as _os_early
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import yaml as _yaml

# On PREEMPT_RT kernels, libk4a's device threads must be created before
# scipy/franky/numpy-BLAS set PTHREAD_PRIO_INHERIT on mutexes. Open and
# start all Kinects here, stash the handles, and pass them to the pipeline
# so it never has to call PyK4A.start() after the heavy libs load.
_EARLY_KINECTS = {}


def _early_kinect_settings():
    """
    Read the kinect_* keys for the launched family straight from
    configs/<family>_default.yaml. Heavy spark_real imports are off-limits
    this early, so this parses argv and the yaml directly; defaults mirror
    PipelineConfig. Per-machine overlays are not consulted here; if one
    ever overrides kinect_* keys this must learn to merge it.

    Returns None when argv names no --robot/--family: no Kinect is opened
    (--robot is required in config.py and main() rejects such a launch).
    """
    fam = None
    argv = sys.argv
    for _j, _a in enumerate(argv):
        if _a in ("--robot", "--family") and _j + 1 < len(argv):
            fam = argv[_j + 1]
        elif _a.startswith(("--robot=", "--family=")):
            fam = _a.split("=", 1)[1]
    if fam is None:
        return None
    raw = {}
    try:
        p = Path(__file__).parent / "configs" / f"{fam}_default.yaml"
        raw = _yaml.safe_load(p.read_text()) or {}
    except Exception:
        pass
    # Fallbacks MUST mirror PipelineConfig (720P / WFOV_2X2BINNED / 15). This
    # early path cannot import pipeline_types' values without pulling heavy
    # deps in before libk4a, so the numbers are duplicated -- and
    # test_usb_bandwidth_budget.py asserts the two stay in agreement. A
    # config missing these keys must NOT silently fall back to the
    # max-bandwidth profile that this host dies under.
    return {
        "resolution": str(raw.get("kinect_resolution", "720P")),
        "depth_mode": str(raw.get("kinect_depth_mode", "WFOV_2X2BINNED")),
        "fps": int(raw.get("kinect_fps", 15)),
        "use_hw_sync": bool(raw.get("kinect_use_hw_sync", False)),
        # YAML-first: the ur10e and franka configs carry an explicit
        # kinect_master_serial (ur10e=000000000000, franka=000000000000);
        # bimanual_franka and g1 have none. The literal below is the FR3
        # host's sideview Kinect serial, kept only as a last-resort fallback
        # if the key is absent from the loaded config.
        "master_serial": str(raw.get("kinect_master_serial", "000000000000")),
    }


def _early_usb_preflight(settings, n_devices):
    """
    Refuse an over-budget camera configuration BEFORE any device is opened.

    An over-subscribed profile starts fine and kills the host an hour later,
    with no panic and no oops. Import is cheap (stdlib + spark_real.usb_budget
    only), so it is safe here, ahead of libk4a's device threads. Set
    SPARK_USB_BUDGET=warn to run the old profile deliberately
    (docs/MULTICAM_CRASH_MECHANISM.md).
    """
    from spark_real import usb_budget as _ub

    loads = [
        _ub.kinect_load(
            f"kinect{i}",
            settings["resolution"],
            settings["depth_mode"],
            settings["fps"],
        )
        for i in range(n_devices)
    ]
    # The wrist RealSense shares the controller even though it is opened
    # later by the pipeline; count it here or the budget lies.
    loads.append(_ub.realsense_load("wrist realsense (color-only)"))
    return _ub.preflight(loads, log=logging.getLogger("spark_server"))



# ---------------------------------------------------------------------------
# SINGLE-INSTANCE GUARD + USB SETTLE. Must run BEFORE the pyk4a block below,
# because that block touches the Kinects at import time.
#
# The port is the only thing that fails fast when a previous instance is still
# alive. Without this guard a restart races the old instance's USB teardown
# with the new instance's Kinect opens, and the host has hard-hung within
# seconds of that overlap (final kernel lines: camera USB traffic).
#
# Binding first turns "two owners on the bus" into "Address already in use"
# before a single USB packet moves. The settle delay covers the tail the bind
# cannot see: the old process releases the port at os._exit(), but the kernel
# is still reaping its in-flight bulk URBs for a moment after.
# ---------------------------------------------------------------------------
_PREBOUND_SOCKET = None
_KINECT_SETTLE_S = 8.0
_RELEASE_STAMP = Path(__file__).parent / "output" / ".kinect_release_stamp"


def _write_kinect_release_stamp():
    """Timestamp the moment our Kinect handles were released, so the NEXT
    launch knows how fresh the USB teardown is and can wait out the remainder
    of the settle window. Failure to write must never block an exit."""
    try:
        _RELEASE_STAMP.parent.mkdir(parents=True, exist_ok=True)
        _RELEASE_STAMP.write_text(str(time.time()))
    except Exception:
        pass


def _early_port_from_argv(default=8888):
    """--port from argv, parsed here because this runs at import time, long
    before main()'s real CLI parsing. Unparseable values fall back to the
    default rather than crash a launch over a typo."""
    av = sys.argv
    for i, a in enumerate(av):
        if a == "--port" and i + 1 < len(av):
            try:
                return int(av[i + 1])
            except ValueError:
                return default
        if a.startswith("--port="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _prebind_and_settle():
    global _PREBOUND_SOCKET
    if _os_early.environ.get("SPARK_NO_PREBIND"):
        return  # escape hatch for diagnostics that import this module
    # Only the actual server launch claims the port. Anything that merely
    # IMPORTS this module -- pytest collecting routes, a diagnostic script --
    # must not: the guard would either steal the port from a running server or
    # abort the import with SystemExit after a 60 s wait because the live
    # server legitimately holds it.
    if __name__ != "__main__" or "pytest" in sys.modules:
        return
    import socket as _socket

    port = _early_port_from_argv()
    waited = False
    deadline = time.time() + 60.0
    while True:
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        # Same option uvicorn sets, so a socket in TIME_WAIT does not
        # false-positive as a live instance.
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.listen(128)
            break
        except OSError:
            sock.close()
            if time.time() > deadline:
                print(
                    f"FATAL: port {port} still held after 60s -- a previous "
                    "instance is alive. Refusing to touch any USB device.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not waited:
                print(
                    f"port {port} held by the previous instance; waiting for "
                    "it to exit before touching any USB device...",
                    file=sys.stderr,
                )
            waited = True
            time.sleep(0.5)
    _PREBOUND_SOCKET = sock

    # How long the old teardown needs is invisible from here, so wait the
    # settle window minus however long ago the old instance stamped its
    # release. Having had to wait for the port at all means the old process
    # exited moments ago: take the full window.
    wait_s = _KINECT_SETTLE_S if waited else 0.0
    try:
        age = time.time() - _RELEASE_STAMP.stat().st_mtime
        if 0 <= age < _KINECT_SETTLE_S:
            wait_s = max(wait_s, _KINECT_SETTLE_S - age)
    except OSError:
        pass
    if wait_s > 0:
        print(
            f"waiting {wait_s:.1f}s for the previous instance's USB teardown "
            "to settle before opening the Kinects...",
            file=sys.stderr,
        )
        time.sleep(wait_s)


_prebind_and_settle()


try:
    import pyk4a as _k4a

    _es = _early_kinect_settings()
    if _es is None:
        # Launched without --robot. There is no family whose Kinect settings
        # could be opened, and main() rejects the launch anyway: leave the
        # devices alone, same as a host without pyk4a.
        raise ImportError("no --robot on the command line")
    _early_usb_preflight(_es, _k4a.connected_device_count())
    _res_map = {
        "720P": _k4a.ColorResolution.RES_720P,
        "1080P": _k4a.ColorResolution.RES_1080P,
        "1440P": _k4a.ColorResolution.RES_1440P,
        "2160P": _k4a.ColorResolution.RES_2160P,
    }
    _depth_map = {
        "NFOV_UNBINNED": _k4a.DepthMode.NFOV_UNBINNED,
        "NFOV_2X2BINNED": _k4a.DepthMode.NFOV_2X2BINNED,
        "WFOV_UNBINNED": _k4a.DepthMode.WFOV_UNBINNED,
        "WFOV_2X2BINNED": _k4a.DepthMode.WFOV_2X2BINNED,
    }
    _fps_map = {5: _k4a.FPS.FPS_5, 15: _k4a.FPS.FPS_15, 30: _k4a.FPS.FPS_30}

    # Serial map first (open without starting) so hw-sync can arm the
    # subordinate before the master starts firing, which the SDK requires.
    # Cached per boot: serials cannot change while the machine is up, and
    # every Kinect open/close cycle is a risk on this bus (see the instance
    # guard above). Keyed on boot_id because enumeration order is stable
    # within a boot but not across boots or replugs; a replug mid-boot comes
    # with a device-count change. If you swap two Kinects on the same ports
    # mid-boot, delete the cache file.
    def _probe_kinect_serials(n):
        found = {}
        for _i in range(n):
            try:
                _d = _k4a.PyK4A(device_id=_i)
                _d.open()
                found[_i] = _d.serial
                _d.close()
            except Exception:
                pass
        return found

    def _cached_kinect_serials():
        import json as _json

        n = _k4a.connected_device_count()
        cache = Path(__file__).parent / "output" / "kinect_serial_cache.json"
        try:
            boot_id = Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text().strip()
        except OSError:
            boot_id = ""
        if boot_id:
            try:
                c = _json.loads(cache.read_text())
                if c.get("boot_id") == boot_id and c.get("count") == n:
                    return {int(k): v for k, v in c["serials"].items()}
            except Exception:
                pass
        found = _probe_kinect_serials(n)
        if boot_id and found:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(
                    _json.dumps(
                        {"boot_id": boot_id, "count": n, "serials": found}
                    )
                )
            except Exception:
                pass
        return found

    _serials = _cached_kinect_serials()

    _use_sync = (
        _es["use_hw_sync"]
        and len(_serials) >= 2
        and _es["master_serial"] in _serials.values()
    )
    _order = sorted(_serials)
    if _use_sync:
        _order.sort(key=lambda i: _serials[i] == _es["master_serial"])

    for _i in _order:
        if _use_sync:
            _mode = (
                _k4a.WiredSyncMode.MASTER
                if _serials[_i] == _es["master_serial"]
                else _k4a.WiredSyncMode.SUBORDINATE
            )
        else:
            _mode = _k4a.WiredSyncMode.STANDALONE
        # WFOV laser pulses are ~10x NFOV's, so subordinates need a larger
        # phase offset to avoid cross-illumination (matches the delays in
        # pipeline_init._init_kinects).
        _delay = 0
        if _mode == _k4a.WiredSyncMode.SUBORDINATE:
            _delay = 1600 if _es["depth_mode"].startswith("WFOV") else 160
        try:
            _d = _k4a.PyK4A(
                _k4a.Config(
                    color_resolution=_res_map.get(
                        _es["resolution"], _k4a.ColorResolution.RES_720P
                    ),
                    depth_mode=_depth_map.get(
                        _es["depth_mode"], _k4a.DepthMode.WFOV_2X2BINNED
                    ),
                    camera_fps=_fps_map.get(_es["fps"], _k4a.FPS.FPS_15),
                    synchronized_images_only=True,
                    wired_sync_mode=_mode,
                    subordinate_delay_off_master_usec=_delay,
                ),
                device_id=_i,
            )
            _d.start()
            _EARLY_KINECTS[_i] = _d
        except Exception as _e:
            print(
                f"[early-kinect] device {_i} failed: {type(_e).__name__}: {_e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            if _use_sync:
                # A half-built sync chain leaves an armed subordinate
                # waiting on triggers that never come. Release everything
                # and let pipeline_init reopen with its full retry logic.
                for _dd in _EARLY_KINECTS.values():
                    for _op in ("stop", "close"):
                        try:
                            getattr(_dd, _op)()
                        except Exception:
                            pass
                _EARLY_KINECTS = {}
                break
except ImportError:
    pass

_os_early.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os_early.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
# SAM3 weights are cached locally; without this, hf_hub pings huggingface.co
# on every start to recheck them. Export HF_HUB_OFFLINE=0 to fetch updates.
_os_early.environ.setdefault("HF_HUB_OFFLINE", "1")

import contextlib as _contextlib

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("spark_server")

# Post-mortem visibility: libfranka/libk4a C++ terminate and unraised thread
# exceptions leave no traceback. faulthandler dumps all thread stacks on a
# fatal signal to stderr (captured by the launch script's log redirect), and
# threading.excepthook records uncaught exceptions in worker threads.
import faulthandler as _faulthandler  # noqa: E402
import threading as _threading_hook  # noqa: E402

try:
    _faulthandler.enable(file=sys.stderr)
except Exception as _fh_exc:  # noqa: BLE001 - never block boot on diagnostics
    logger.warning("faulthandler.enable failed: %s", _fh_exc)


def _log_thread_exception(args):
    logger.critical(
        "UNCAUGHT EXCEPTION in thread %r",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


_threading_hook.excepthook = _log_thread_exception

# Add SAM3 / DA3 to path
for p in [Path.home() / "mv_sam3" / "sam3"]:
    if p.exists():
        sys.path.insert(0, str(p))
        break
for p in [Path.home() / "mv_sam3"]:
    if p.exists():
        sys.path.insert(0, str(p))
        break


# Hot-reload (jurigged)
# Watch spark_real/**/*.py and live-patch functions/classes on save so the
# user can iterate without restarting (a restart pays the SAM3/EquiGraspFlow/
# pyroki warm-up cost and risks the Kinect MCU-brick path on shutdown).
# Set SPARK_HOT_RELOAD=0 to disable. Must run BEFORE pipeline import so
# later module loads are registered with jurigged.
if _os_early.environ.get("SPARK_HOT_RELOAD", "1") != "0":
    try:
        import jurigged

        # jurigged uses fnmatch (not glob), so `**` is NOT recursive, it's
        # just two literal `*`s. Workaround: pass a LIST of two patterns,
        # one for top-level .py files, one for nested. Together they cover
        # every .py file under spark_real. jurigged's auto_register only
        # ever calls the filter on sys.modules entries (all real .py files),
        # so non-.py files in output/, calibrations/, videos/ are never
        # offered to the filter and never watched.
        _spark_real_root = str(Path(__file__).resolve().parent)
        jurigged.watch(
            [
                f"{_spark_real_root}/*.py",
                f"{_spark_real_root}/**/*.py",
            ]
        )
        logger.info(
            "[hot-reload] jurigged watching spark_real "
            "(set SPARK_HOT_RELOAD=0 to disable)"
        )
    except Exception as e:
        logger.warning("[hot-reload] jurigged unavailable: %s", e)


# App setup

app = FastAPI(title="SPARK Real Pipeline", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Progress logging

from spark_real.routes import state


class _ProgressHandler(logging.Handler):
    """
    Captures pipeline log messages for the frontend progress endpoint.

    Stamps each record with the monotonic sequence number the frontend
    polls against, and with record.created (when the event occurred) rather
    than the handling time.
    """

    def emit(self, record):
        msg = self.format(record)
        with state.progress_lock:
            state.progress_seq += 1
            state.progress_log.append(
                {"seq": state.progress_seq, "ts": record.created, "msg": msg}
            )
            # Trim from the front. Safe now that clients track `seq`
            # rather than a list index.
            excess = len(state.progress_log) - state.PROGRESS_MAX
            if excess > 0:
                del state.progress_log[:excess]


# ATTACH AT THE PACKAGE ROOT, NOT PER-MODULE. A per-leaf allowlist rots
# whenever a module is split (each mixin takes getLogger(__name__)), and the
# symptom is a /api/progress buffer that never grows while a 133-second run
# executes in silence. Propagation from the "spark_real" root carries records
# from any spark_real.* logger, including ones not yet written. Measured cost
# on a 113-second pick-and-place: ~76 records, under one per second.
# "spark_server" is a separate root and needs its own.
for _logger_name in ["spark_real", "spark_server"]:
    _prog_handler = _ProgressHandler()
    _prog_handler.setLevel(logging.INFO)
    _prog_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger(_logger_name).addHandler(_prog_handler)


# Register route modules

from spark_real.routes.core import router as core_router
from spark_real.routes.control import router as control_router
from spark_real.routes.calibration import router as calibration_router
from spark_real.routes.streaming import router as streaming_router
from spark_real.routes.episode import router as episode_router
from spark_real.routes.annotate_trace import router as trace_router

# Bimanual routes mount under /api/bimanual; safe to include on every
# build because each handler 400s when the active family is not bimanual.
from spark_real.routes.bimanual import router as bimanual_router

# Tactile routes 200 with ``connected: false`` when no FlexiTac sensor
# is plugged in, so it's safe to mount unconditionally.
from spark_real.routes.tactile import router as tactile_router

# Subprocess-managed viser-IK teleop (external robots_realtime package).
# Endpoints 400 cleanly when the uv venv is not installed.
from spark_real.routes.viser_teleop import router as viser_teleop_router

# Subprocess-managed OSC executor backend (same external package).
# Provides an A/B-testable panda-py + OSC alternative to the franky path.
from spark_real.routes.osc_executor import router as osc_executor_router

# BT library browser + cache/LLM toggle (runtime override for
# SPARK_DISABLE_LLM) + single-shot BT pin for /api/execute.
from spark_real.routes.bt_library import router as bt_library_router

# RF-DETR second-opinion switch (auto / always / off) + read-only fusion
# gate status. Runtime override for perception.fusion.proposer; drops the
# pipeline's cached gate so a toggle applies to the very next detect.
from spark_real.routes.perception import router as perception_router

# Perception-grounded auto-MJCF scene generator + sim preview.
# Converts /api/detect_approve detections into a MuJoCo XML for
# sim-before-real safety gating. No hardware deps.
from spark_real.routes.auto_scene import router as auto_scene_router
from spark_real.routes.trials import router as trials_router

# Task-execution scores across the different BTs the planner generated,
# grouped by instruction. Read-only aggregation of output/real_runs; serves
# the /scores page. No hardware deps.
from spark_real.routes.bt_scores import router as bt_scores_router

# Demonstration recording (/api/vla_record/*). Writes synchronized
# image + proprio + commanded-action episodes to a data root outside the
# repo, in the SAME on-disk schema as the human teleop corpus, so demos
# generated by SPARK's executor and demos driven by an operator are
# interchangeable training data. Recorder only READS robot/camera state;
# numpy+opencv, no heavy deps at import.
from spark_real.routes.vla_record import router as vla_record_router

app.include_router(core_router)
app.include_router(control_router)
app.include_router(calibration_router)
app.include_router(streaming_router)
app.include_router(episode_router)
app.include_router(trace_router)
app.include_router(bimanual_router)
app.include_router(tactile_router)
app.include_router(viser_teleop_router)
app.include_router(osc_executor_router)
app.include_router(bt_library_router)
app.include_router(perception_router)
app.include_router(auto_scene_router)
app.include_router(trials_router)
app.include_router(bt_scores_router)
app.include_router(vla_record_router)


import atexit as _atexit
import signal as _signal
import threading as _threading

import tyro

from spark_real.config import SparkConfig, load_profile
from spark_real.pipeline import SPARKRealPipeline, PipelineConfig
from spark_real.routes.calibration import (
    load_saved_anchor,
    load_saved_anchor_points,
)

# Optional Franka Desk auto-unlock and pyroki IK warmup. Both are only
# exercised on the Franka launch path; wrapped so the server imports on
# hosts without libfranka / pyroki present.
try:
    from spark_real.robots.franka.desk_session import auto_unlock_and_enable_fci
except Exception:
    auto_unlock_and_enable_fci = None

try:
    from spark_real.control.fr3_ik_pyroki import warmup as _ik_warmup
except Exception:
    _ik_warmup = None

# One-shot latch, not a bool: the callers are a signal handler
# (SIGTERM/SIGINT/SIGHUP), uvicorn's on_event("shutdown"), and atexit, and a
# signal can interrupt the main thread between a check and a set, re-entering
# the whole teardown. Acquired once and never released, so every later caller
# (including a reentrant signal) returns immediately. A Lock held across the
# body would instead deadlock on that same reentry.
_shutdown_latch = _threading.Lock()
_SHUTDOWN_BUDGET_S = 10.0  # hard deadline, after which we os._exit


def _release_devices(reason: str):
    """
    Idempotent cleanup of Kinects, RealSense, robot. Safe to call from
    signal handlers and atexit. Closing Kinects without this leaves the
    depth MCU mid-stream and bricks it for the next launch.

    Runs pipeline.shutdown() in a worker thread with a hard deadline.
    Past the deadline, we abandon whatever step is hung (most often
    libfranka disconnect): the cameras' own close() ran first and
    each has its own internal watchdog, so they release before robot
    disconnect even gets a chance to wedge.
    """
    if not _shutdown_latch.acquire(blocking=False):
        return

    # BRAKE THE ARM FIRST. Nothing below this may run before it: the brake used
    # to sit inside robot.disconnect(), behind subprocess teardown and three
    # camera closes, and was skipped entirely when pipeline.shutdown() blew its
    # deadline -- so Ctrl-C could exit with a movej still running and the arm
    # coasting to its target. One URScript send on the already-open socket,
    # bounded, and every failure swallowed: a stuck camera must never be able
    # to prevent the stop.
    try:
        _pipe = state.pipeline
        _rb = getattr(_pipe, "_robot", None) if _pipe is not None else None
        if _rb is not None and getattr(_rb, "SUPPORTS_URSCRIPT", False):
            _send = getattr(_rb, "_send_script", None)
            if callable(_send):
                _send("stopj(3.0)")
                logger.warning("shutdown: braked the arm (%s)", reason)
    except Exception as _exc:  # noqa: BLE001 - shutdown must continue regardless
        logger.warning("shutdown: brake failed: %s", _exc)

    # STOP THE DEMO RECORDER NEXT, before anything is closed. Its loop calls
    # pipeline.capture() and reads proprio over RTDE at 15 Hz; a
    # get_observation() racing robot.disconnect() segfaults inside ur_rtde
    # rather than raising. discard() rather than end(): a half-recorded
    # episode killed by a signal is not training data, and end() would try to
    # encode a video during a bounded shutdown.
    _rec = getattr(state, "vla_recorder", None)
    if _rec is not None:
        try:
            _rec.discard()
            logger.info("shutdown: stopped the demo recorder (%s)", reason)
        except Exception as _exc:  # noqa: BLE001
            logger.warning("shutdown: demo recorder stop failed: %s", _exc)
        try:
            state.vla_recorder = None
            state.vla_recording = False
        except Exception:  # noqa: BLE001
            pass

    # Then the stream/websocket capture pool. Its workers sit inside
    # capture_single_camera and would otherwise keep reading cameras that the
    # shutdown below is releasing. cancel_futures drops the queued backlog;
    # wait=False so a worker blocked in a camera read cannot consume the
    # shutdown budget on its own.
    try:
        from spark_real.routes.streaming import _STREAM_POOL

        _STREAM_POOL.shutdown(wait=False, cancel_futures=True)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("shutdown: stream pool shutdown failed: %s", _exc)

    # Kill subprocess-managed children (OSC, viser-teleop) BEFORE touching
    # the pipeline. They were spawned with start_new_session=True so they
    # would otherwise outlive the parent and an orphan keeps port 5555 bound,
    # blocking the next /api/control/osc/start. SIGTERM their process group
    # then SIGKILL on timeout. Best-effort; never blocks the rest of shutdown.
    for attr in ("osc_executor_proc", "viser_teleop_proc"):
        proc = getattr(state, attr, None)
        if proc is None or proc.poll() is not None:
            continue
        try:
            _os_early.killpg(_os_early.getpgid(proc.pid), _signal.SIGTERM)
            logger.info(
                "shutdown: SIGTERM %s pgid=%d", attr, _os_early.getpgid(proc.pid)
            )
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                _os_early.killpg(_os_early.getpgid(proc.pid), _signal.SIGKILL)
                logger.info("shutdown: SIGKILL %s (didn't exit on TERM)", attr)
            except Exception:
                pass
        try:
            setattr(state, attr, None)
        except Exception:
            pass

    pipe = state.pipeline
    if pipe is None:
        return
    logger.info("releasing devices (%s)", reason)

    def _do():
        try:
            pipe.shutdown()
        except Exception as exc:
            logger.warning("pipeline shutdown raised: %s", exc)

    t = _threading.Thread(target=_do, daemon=True, name="spark-shutdown")
    t.start()
    t.join(timeout=_SHUTDOWN_BUDGET_S)
    if t.is_alive():
        logger.warning(
            "shutdown exceeded %.0fs, proceeding to exit anyway. "
            "Cameras should already be released (per-device watchdogs); "
            "robot disconnect was likely the holdup.",
            _SHUTDOWN_BUDGET_S,
        )
    # Whichever way shutdown went, the Kinect handles are gone NOW. Stamp it
    # so the next launch can wait out the kernel's URB reaping (see
    # _prebind_and_settle) instead of opening into the middle of it.
    _write_kinect_release_stamp()


@app.on_event("shutdown")
def _on_shutdown():
    """
    FastAPI/Uvicorn graceful path (no signal involved, or SIGTERM
    drained through Uvicorn's signal trap).
    """
    _release_devices("uvicorn shutdown")


def _signal_handler(signum, _frame):
    """
    Catch SIGTERM/SIGINT before Uvicorn's loop can stall on a busy
    pipeline. Calls shutdown synchronously then exits via os._exit so
    we never get stuck waiting for the event loop to drain. SIGKILL
    bypasses this: use scripts/spark_server.sh stop, never `kill -9`.
    """
    name = _signal.Signals(signum).name
    try:
        _release_devices(f"signal {name}")
    finally:
        _os_early._exit(0)


_signal.signal(_signal.SIGTERM, _signal_handler)
_signal.signal(_signal.SIGINT, _signal_handler)
_signal.signal(_signal.SIGHUP, _signal_handler)
_atexit.register(_release_devices, "atexit")


# Static files + frontend

FRONTEND_DIR = Path(__file__).parent / "frontend"
VIDEOS_DIR = Path(__file__).parent / "output" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return HTMLResponse(
            (FRONTEND_DIR / "index.html").read_text(),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )

else:

    @app.get("/")
    async def serve_placeholder():
        return HTMLResponse("<h1>SPARK Real Pipeline</h1><p>Frontend not built.</p>")


# Convenience accessor (used by pipeline)


def main():
    # One tyro CLI over the shared SparkConfig. --family (aliased to the
    # historical --robot) plus --host/--port/--ip and the --no-robot/
    # --no-init/--no-kinect/--auto-unlock/--strict-verify flags all derive
    # from the dataclass fields, so existing launch scripts keep working.
    cfg = tyro.cli(SparkConfig)
    # Remember the launched config so /api/initialize can rebuild THE SAME
    # family's pipeline instead of a default-family (franka) one on the
    # UR10e rig.
    state.launch_config = cfg

    if not cfg.no_init:
        # Resolve the robot setup from configs/<family>_default.yaml (plus an
        # optional per-machine overlay) through the shared loader in
        # spark_real/config.py. tests/test_config_equivalence.py pins this.
        profile = load_profile(cfg)
        # Resolved IP is reused below by the Franka auto-unlock path.
        ip_str = profile.resolved_ip()
        config = PipelineConfig(**profile.to_pipeline_kwargs())

        # Franka auto-unlock: do this BEFORE pipeline.initialize() because
        # initialize() will construct franky.Robot, which fails if FCI
        # isn't active. Family-gated so UR10e/G1 startups skip it.
        if cfg.auto_unlock and cfg.family == "franka" and not cfg.no_robot:
            try:
                if auto_unlock_and_enable_fci is None:
                    raise ImportError("desk_session unavailable")
                ok = auto_unlock_and_enable_fci(ip_str)
                logger.info(
                    "Desk auto-unlock %s for %s",
                    "succeeded" if ok else "did not complete",
                    ip_str,
                )
            except Exception as _e:
                logger.warning("Desk auto-unlock raised: %s (continuing)", _e)
        elif cfg.auto_unlock and cfg.family != "franka":
            logger.info("--auto-unlock is Franka-only; ignored for %s", cfg.family)

        try:
            state.pipeline = SPARKRealPipeline(config, profile=profile)
            state.pipeline._pre_opened_kinects = _EARLY_KINECTS
            logger.info(
                "Early Kinects passed to pipeline: %s",
                (
                    {k: v.serial for k, v in _EARLY_KINECTS.items()}
                    if _EARLY_KINECTS
                    else "none"
                ),
            )
            state.pipeline.initialize()
        except BaseException:
            # A failed boot must not abandon the early-opened Kinects
            # mid-stream: atexit's _release_devices only walks
            # pipeline-owned devices, and a depth MCU left streaming is the
            # documented brick path (12V power-cycle before the next
            # launch). Stop + close them before the exception propagates.
            logger.critical(
                "Pipeline construction/initialize failed; releasing %d "
                "early-opened Kinect(s) before exiting",
                len(_EARLY_KINECTS),
            )
            for _d in _EARLY_KINECTS.values():
                for _op in ("stop", "close"):
                    try:
                        getattr(_d, _op)()
                    except Exception:  # noqa: BLE001
                        pass
            _EARLY_KINECTS.clear()
            _write_kinect_release_stamp()
            raise
        load_saved_anchor(state.pipeline)
        load_saved_anchor_points()

        if cfg.family == "franka" and not cfg.no_robot:
            # JIT-compile the IK off the critical path so uvicorn can start
            # serving immediately. JAX serializes compilation internally, so
            # an IK call racing the warmup just blocks on the same compile.
            def _warm_ik():
                try:
                    if _ik_warmup is None:
                        raise ImportError("pyroki IK unavailable")
                    logger.info("Warming up pyroki IK (one-time JIT compile)...")
                    t0 = time.time()
                    _ik_warmup()
                    logger.info("pyroki IK warmup done in %.1fs", time.time() - t0)
                except Exception as _e:
                    logger.warning("pyroki IK warmup raised: %s", _e)

            _threading.Thread(target=_warm_ik, name="ik-warmup", daemon=True).start()

    # Run the server WITHOUT uvicorn's signal handling. uvicorn.run() installs
    # its own SIGINT/SIGTERM handlers, which replace the module-level
    # _signal_handler above -- and uvicorn's graceful path waits for in-flight
    # requests to drain. A long /api/execute sitting in the servo loop (e.g.
    # re-hitting "CBF-QP infeasible") never returns, so Ctrl-C would hang with
    # no way out but kill -9. Keeping our handler means Ctrl-C releases the
    # devices and os._exit()s within the shutdown budget.
    class _KeepOurSignalHandlers(uvicorn.Server):
        def capture_signals(self):
            return _contextlib.nullcontext()

    _srv = _KeepOurSignalHandlers(
        uvicorn.Config(app, host=cfg.host, port=cfg.port)
    )
    # Serve on the socket the import-time guard already bound, so the
    # fail-fast port claim and the port actually served are one and the same
    # -- no unbind/rebind window for a second instance to slip through. If
    # the real config resolved a different port than the guard's argv parse
    # (guard saw only --port), fall back to a plain bind on the right port.
    if (
        _PREBOUND_SOCKET is not None
        and _PREBOUND_SOCKET.getsockname()[1] == cfg.port
    ):
        _srv.run(sockets=[_PREBOUND_SOCKET])
    else:
        if _PREBOUND_SOCKET is not None:
            logger.warning(
                "prebound port %d != configured port %d; releasing the guard "
                "socket and binding normally",
                _PREBOUND_SOCKET.getsockname()[1],
                cfg.port,
            )
            _PREBOUND_SOCKET.close()
        _srv.run()


if __name__ == "__main__":
    main()
