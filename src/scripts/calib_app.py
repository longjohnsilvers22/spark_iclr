"""Guided EYE-TO-HAND calibration web app: the static Kinects, not the wrist.

Arm-agnostic -- the operator drives the gripper by hand, so nothing here
generates poses or moves the robot. Used on both the UR10e and the FR3.

A browser front-end for the depth-supervised RGB-D Procrustes calibration in
scripts/spark_calibrate.py. It drives the same spark_real server API
(/api/calibrate/*) and writes the same files (/tmp/board_pose.json and the
installed handeye_<cam>.json), but replaces the terminal input() prompts with a
photo you can look at and tap.

What it does for you:
  - Shows the live birdview (or sideview) with every ChArUco corner marked and
    ONE target corner highlighted ("GO HERE"), so you know exactly which corner
    to drive the gripper to.
  - Anchor: drive the gripper to the highlighted corner, press Record. The app
    pairs that corner with the touched TCP (no dependency on the existing
    camera cal), advances to the next spread-out corner, and solves
    T_board->base when you have enough touches.
  - Heights: type the board height (mm) and press Capture, and it grabs RGB+depth
    from ALL static cameras at once and accumulates corner pairs. It tells you
    when the depth spread is too small (raise the board) or the board is near a
    view edge (slide it), then solves and installs handeye_birdview/sideview.

Run on the robot host (server must be up), in the env that has the spark deps:
    cd <repo>/src && python scripts/calib_app.py
Open http://localhost:8892 (or http://<host-ip>:8892).

The wrist (eye-in-hand) is a separate CLI step that DOES auto-drive the arm:
    spark_calibrate.py --mode calibrate_wrist --board-pose /tmp/board_pose.json
"""
import base64
import json
import os
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response

# spark_calibrate.py lives in ~/spark/scripts. This app lives in
# ~/spark/src/scripts. Resolve the sibling repo-root scripts dir from __file__
# so the import works wherever the repo is checked out.
_HERE = Path(__file__).resolve().parent
for cand in (_HERE.parents[1] / "scripts", _HERE):
    if (cand / "spark_calibrate.py").exists():
        sys.path.insert(0, str(cand))
        break
import spark_calibrate as sc  # noqa: E402

STATIC_CAMS = ["birdview", "sideview"]
BOARD_POSE_PATH = Path("/tmp/board_pose.json")
CAL_DIR = sc.INSTALLED_CAL_DIR

app = Flask(__name__)
_lock = threading.Lock()
# Per-camera locks let birdview and sideview live frames fetch in parallel
# (the server has its own per-device locks); _lock still serializes the
# data-collecting operations (record / heights / solves).
_cam_locks = {c: threading.Lock() for c in ("birdview", "sideview", "wrist")}
# Live frames are served at half resolution; /tap scales clicks back up.
DISPLAY_SCALE = 0.5

# Loaded once at startup.
_board, _dictionary, _meta = sc.load_board(sc.DEFAULT_BOARD)
_chess = _board.getChessboardCorners()  # (N, 3) corner xyz in board frame


def _suggested_order():
    # Touch order that spreads across the board: 4 extreme corners, then the
    # center, then the rest. A spread set makes the Procrustes well conditioned.
    xy = _chess[:, :2]
    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    anchors = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax),
               ((xmin + xmax) / 2, (ymin + ymax) / 2)]
    order = []
    for ax, ay in anchors:
        cid = int(np.argmin(np.hypot(xy[:, 0] - ax, xy[:, 1] - ay)))
        if cid not in order:
            order.append(cid)
    for cid in range(len(_chess)):
        if cid not in order:
            order.append(cid)
    return order


STATE = {
    "order": _suggested_order(),
    "target": None,            # current target corner id
    "px_cache": {},            # {cam: {corner_id: (u, v)}} last-seen pixels
    "touches": [],             # [{corner_id, tcp_xyz}]
    "acc": {c: {"m": [], "o3": [], "pix": [], "intr": None, "heights": []}
            for c in STATIC_CAMS},
    "guidance": "Place the board flat, pick a camera, and start the anchor.",
    "anchor_rms": None,
    "install": None,
}
STATE["target"] = STATE["order"][0]

# Every recorded touch is mirrored to disk so an app restart (or crash)
# never loses manually collected corner touches.
TOUCHES_PATH = Path("/tmp/calib_app_touches.jsonl")


def _next_target():
    done = {t["corner_id"] for t in STATE["touches"]}
    for cid in STATE["order"]:
        if cid not in done:
            return cid
    return None


def _persist_touches():
    TOUCHES_PATH.write_text(
        "".join(json.dumps(t) + "\n" for t in STATE["touches"]))


if TOUCHES_PATH.exists():
    for _line in TOUCHES_PATH.read_text().splitlines():
        if _line.strip():
            STATE["touches"].append(json.loads(_line))
    STATE["target"] = _next_target()
    if STATE["touches"]:
        STATE["guidance"] = (f"Restored {len(STATE['touches'])} saved touches "
                             f"from {TOUCHES_PATH}.")


def _capture(cam):
    # (rgb, intr, depth, ids, pixels). Updates the per-corner pixel cache.
    rgb, intr, depth = sc.fetch_frame(cam)
    ids, pixels = sc.detect_charuco_corners(rgb, _board, _dictionary)
    cache = STATE["px_cache"].setdefault(cam, {})
    if ids is not None:
        for cid, (u, v) in zip(ids, pixels):
            cache[int(cid)] = (float(u), float(v))
    return rgb, intr, depth, ids, pixels


def _fetch_rgb(cam):
    # RGB-only capture for the live view: skips the depth round-trip.
    # Detection runs at full resolution: half-res finds only a partial corner
    # set on the oblique sideview and leaves target markers missing there.
    import base64
    import io

    import requests
    from PIL import Image

    r = requests.get(f"{sc.SERVER}/api/calibrate/capture_one",
                     params={"camera": cam}, timeout=10)
    r.raise_for_status()
    rgb = np.array(Image.open(io.BytesIO(
        base64.b64decode(r.json()["rgb_png_base64"]))))
    ids, pixels = sc.detect_charuco_corners(rgb, _board, _dictionary)
    cache = STATE["px_cache"].setdefault(cam, {})
    if ids is not None:
        for cid, (u, v) in zip(ids, pixels):
            cache[int(cid)] = (float(u), float(v))
    STATE.setdefault("corners_seen", {})[cam] = {
        "n": 0 if ids is None else int(len(ids)), "of": len(_chess),
        "at": datetime.now().strftime("%H:%M:%S")}
    return rgb, ids, pixels


def _render(cam, ids, pixels, rgb):
    # Draw all detected corners, the recorded touches, and the target marker.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ids is not None:
        for cid, (u, v) in zip(ids, pixels):
            cv2.circle(bgr, (int(u), int(v)), 4, (80, 220, 80), -1)
    touched = {t["corner_id"] for t in STATE["touches"]}
    cache = STATE["px_cache"].get(cam, {})
    for cid in touched:
        if cid in cache:
            u, v = cache[cid]
            cv2.circle(bgr, (int(u), int(v)), 8, (240, 160, 40), 2)
    tgt = STATE["target"]
    if tgt is not None and tgt in cache:
        u, v = cache[tgt]
        u, v = int(u), int(v)
        cv2.circle(bgr, (u, v), 22, (40, 40, 240), 3)
        cv2.line(bgr, (u - 34, v), (u + 34, v), (40, 40, 240), 2)
        cv2.line(bgr, (u, v - 34), (u, v + 34), (40, 40, 240), 2)
        cv2.putText(bgr, f"GO HERE  corner {tgt}", (u + 28, v - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 240), 2)
    elif tgt is not None:
        cv2.putText(bgr, f"target corner {tgt} not visible from this camera",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 160, 255), 2)
    # Half-res display: halves encode time and transfer size. Taps are
    # scaled back up by DISPLAY_SCALE in /tap.
    bgr = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2))
    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return jpg.tobytes() if ok else None


def _edge_hint(cam, ids, pixels, depth):
    # Tell the user to slide the board (toward view center) or raise it.
    if ids is None or len(ids) == 0:
        return f"{cam}: no ChArUco corners visible, recenter the board."
    h, w = depth.shape[:2]
    px = np.asarray(pixels)
    cx, cy = px[:, 0].mean(), px[:, 1].mean()
    msgs = []
    n_total = len(_chess)
    if len(px) < n_total:
        if cx < w * 0.30:
            msgs.append("slide board RIGHT")
        elif cx > w * 0.70:
            msgs.append("slide board LEFT")
        if cy < h * 0.30:
            msgs.append("slide board DOWN")
        elif cy > h * 0.70:
            msgs.append("slide board UP")
    seen = f"{len(px)}/{n_total} corners"
    return f"{cam}: {seen}" + (", " + ", ".join(msgs) if msgs else "")


# Stale browser tabs from older versions of this page keep their polling
# loops running forever; frames require the current page version so a
# forgotten tab can never trigger camera captures.
PAGE_VERSION = "4"


@app.route("/frame.jpg")
def frame_jpg():
    if request.args.get("v") != PAGE_VERSION:
        return Response("stale page, reload the tab", status=403)
    cam = request.args.get("cam", "birdview")
    # The server's capture path occasionally returns a transient miss (busy
    # device, momentary empty frame); retry a couple of times before failing
    # so the operator does not see broken images for one-off blips.
    last_exc = None
    for attempt in range(3):
        try:
            with _cam_locks.setdefault(cam, threading.Lock()):
                rgb, ids, pixels = _fetch_rgb(cam)
                jpg = _render(cam, ids, pixels, rgb)
            if jpg is None:
                return Response("encode failed", status=500)
            return Response(jpg, mimetype="image/jpeg")
        except Exception as exc:
            last_exc = exc
            time.sleep(0.4)
    return Response(f"capture failed: {last_exc}", status=503)


@app.route("/tap", methods=["POST"])
def tap():
    # Tap on either photo to choose the nearest detected corner as the target.
    d = request.get_json(force=True)
    cam = str(d.get("cam", "birdview"))
    # Clicks arrive in displayed (half-res) coordinates; the corner cache is
    # full-res.
    u = float(d["u"]) / DISPLAY_SCALE
    v = float(d["v"]) / DISPLAY_SCALE
    cache = STATE["px_cache"].get(cam, {})
    if not cache:
        return jsonify(ok=False, error=f"no corners detected on {cam} yet")
    cid = min(cache, key=lambda k: np.hypot(cache[k][0] - u, cache[k][1] - v))
    with _lock:
        STATE["target"] = cid
    return jsonify(ok=True, target=cid)


TELEOP_TCP_PATH = Path("/tmp/teleop_tcp.json")


def _current_tcp():
    """(xyz, source). A fresh teleop-published TCP wins, else the server.

    A teleop holding the robot exclusively (FR3/libfranka) makes the server's
    reads return null, so it publishes FK to TELEOP_TCP_PATH instead. Absent
    file => nobody is hogging the arm, read the server.
    """
    if TELEOP_TCP_PATH.exists():
        try:
            d = json.loads(TELEOP_TCP_PATH.read_text())
            xyz = d.get("tcp_xyz") or []
            if (time.time() - float(d.get("ts", 0))) < 2.0 and len(xyz) == 3:
                return [float(v) for v in xyz], "teleop"
        except Exception:
            pass
    try:
        tcp = sc.fetch_tcp()
        xyz = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
        if all(np.isfinite(xyz)):
            return xyz, "server"
    except Exception:
        pass
    raise RuntimeError(
        "no TCP source: the server returned null and /tmp/teleop_tcp.json is "
        "missing or stale. Either stop the standalone teleop so the server can "
        "read the robot, or restart a teleop that publishes the TCP to that "
        "path (scripts/teleop_fr3.py does).")


@app.route("/record_touch", methods=["POST"])
def record_touch():
    # Record the current TCP as a touch of the target corner, then advance.
    try:
        xyz, src = _current_tcp()
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))
    tgt = STATE["target"]
    if tgt is None:
        return jsonify(ok=False, error="no target corner selected")
    with _lock:
        STATE["touches"] = [t for t in STATE["touches"]
                            if t["corner_id"] != tgt] + \
            [{"corner_id": tgt, "tcp_xyz": xyz}]
        _persist_touches()
        STATE["target"] = _next_target()
        STATE["guidance"] = (f"Recorded corner {tgt} at "
                             f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) "
                             f"via {src}. {len(STATE['touches'])} touches.")
    return jsonify(ok=True, corner=tgt, tcp_xyz=xyz, target=STATE["target"],
                   n=len(STATE["touches"]))


@app.route("/solve_anchor", methods=["POST"])
def solve_anchor():
    touches = STATE["touches"]
    if len(touches) < 3:
        return jsonify(ok=False, error=f"need >=3 touches, have {len(touches)}")
    ids = [t["corner_id"] for t in touches]
    tcp_pts = np.array([t["tcp_xyz"] for t in touches])

    # Robust fit: iteratively drop the worst touch while it is a clear
    # outlier (residual above 3 mm AND 2.5x the median), keeping >=4. The
    # operator can over-collect touches and let the solve self-clean.
    use = list(range(len(ids)))
    dropped = []
    while True:
        B = _chess[[ids[i] for i in use]]
        P = tcp_pts[use]
        R, t = sc.get_rigid_transform(B, P)
        resid = np.linalg.norm((R @ B.T).T + t - P, axis=1)
        if len(use) <= 4:
            break
        w = int(np.argmax(resid))
        if resid[w] <= max(0.003, 2.5 * float(np.median(resid))):
            break
        dropped.append({"corner_id": ids[use[w]],
                        "residual_mm": round(float(resid[w]) * 1000, 1)})
        use.pop(w)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    rms_mm = float(np.sqrt(np.mean(resid ** 2)) * 1000)
    BOARD_POSE_PATH.write_text(json.dumps({
        "T_board2base": T.tolist(),
        "touches": [{"corner_id": int(ids[i]),
                     "tcp_xyz": list(map(float, tcp_pts[i]))} for i in use],
        "dropped_touches": dropped,
        "rms_mm": rms_mm,
        "board_meta": _meta,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    drop_note = (" Dropped " + ", ".join(
        f"corner {d['corner_id']} ({d['residual_mm']}mm)" for d in dropped)
        + "." if dropped else "")
    with _lock:
        STATE["anchor_rms"] = rms_mm
        STATE["guidance"] = (f"Board anchored: RMS {rms_mm:.1f} mm over "
                             f"{len(use)} touches.{drop_note} "
                             f"Now do the height captures.")
    return jsonify(ok=True, rms_mm=rms_mm,
                   used=[int(ids[i]) for i in use],
                   per_touch_mm=[round(float(e) * 1000, 1) for e in resid],
                   dropped=dropped,
                   saved=str(BOARD_POSE_PATH))


@app.route("/capture_height", methods=["POST"])
def capture_height():
    if not BOARD_POSE_PATH.exists():
        return jsonify(ok=False, error="anchor the board first")
    d = request.get_json(force=True)
    try:
        h = float(d["height_mm"]) / 1000.0
    except Exception:
        return jsonify(ok=False, error="height_mm must be a number")
    T_board2base = np.array(json.loads(BOARD_POSE_PATH.read_text())["T_board2base"])
    result = {}
    hints = []
    try:
        with _lock:
            for cam in STATIC_CAMS:
                rgb, intr, depth, ids, pixels = _capture(cam)
                acc = STATE["acc"][cam]
                acc["intr"] = intr
                n0 = len(acc["m"])
                if ids is not None and depth is not None:
                    for cid, (u, v) in zip(ids, pixels):
                        z = sc._depth_median_at(depth, u, v)
                        if z is None:
                            continue
                        p_board = _chess[int(cid)]
                        p_base = (T_board2base @ np.append(p_board, 1.0))[:3] \
                            + np.array([0.0, 0.0, h])
                        acc["m"].append(p_base)
                        acc["o3"].append([(u - intr["cx"]) * z / intr["fx"],
                                          (v - intr["cy"]) * z / intr["fy"], z])
                        acc["pix"].append([u, v])
                    acc["heights"].append(h)
                added = len(acc["m"]) - n0
                zspread = 0.0
                if acc["o3"]:
                    zc = np.array(acc["o3"])[:, 2]
                    zspread = float(zc.max() - zc.min())
                result[cam] = {"added": added, "total": len(acc["m"]),
                               "z_spread_mm": round(zspread * 1000, 1),
                               "heights_mm": sorted({round(x * 1000) for x in acc["heights"]})}
                hints.append(_edge_hint(cam, ids, pixels, depth))
                if zspread < 0.05:
                    hints.append(f"{cam}: depth spread {zspread*1000:.0f} mm < 50 mm, "
                                 f"raise the board on a riser and capture again.")
        with _lock:
            STATE["guidance"] = " | ".join(hints)
        return jsonify(ok=True, height_mm=round(h * 1000), cams=result, hints=hints)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc))


@app.route("/solve_static", methods=["POST"])
def solve_static():
    anchor_rms = None
    if BOARD_POSE_PATH.exists():
        anchor_rms = json.loads(BOARD_POSE_PATH.read_text()).get("rms_mm")
    out = {}
    for cam in STATIC_CAMS:
        acc = STATE["acc"][cam]
        if len(acc["m"]) < 6:
            out[cam] = {"ok": False, "error": f"only {len(acc['m'])} pairs, need >=6"}
            continue
        m = np.array(acc["m"])
        o3 = np.array(acc["o3"])
        pix = np.array(acc["pix"])
        T_base2cam, scale, offset, rmse = sc.fit_with_depth_scale(m, o3, pix, acc["intr"])
        T_cam2base = np.linalg.inv(T_base2cam)
        dst = CAL_DIR / f"handeye_{cam}.json"
        if dst.exists():
            bak = CAL_DIR / f"handeye_{cam}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.json"
            shutil.copy2(dst, bak)
        dst.write_text(json.dumps({
            "camera": cam,
            "mode": "eye_to_hand_rgbd_procrustes",
            "T_cam_to_base_4x4": T_cam2base.tolist(),
            "depth_scale_correction": float(scale),
            "depth_offset_correction": float(offset),
            "heights_m": sorted({round(x, 4) for x in acc["heights"]}),
            "rmse_mm": float(rmse * 1000),
            "num_pairs": int(len(m)),
            "board_anchor_rms_mm": anchor_rms,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
        out[cam] = {"ok": True, "rmse_mm": round(float(rmse) * 1000, 2),
                    "depth_scale": round(float(scale), 5),
                    "depth_offset_mm": round(float(offset) * 1000, 1),
                    "num_pairs": int(len(m)), "saved": str(dst)}
    with _lock:
        STATE["install"] = out
        STATE["guidance"] = ("Installed. Restart the server "
                             "(spark_server.sh restart) to apply the depth "
                             "correction.")
    return jsonify(ok=True, cams=out)


@app.route("/reset", methods=["POST"])
def reset():
    what = request.get_json(force=True).get("what", "anchor")
    with _lock:
        if what == "anchor":
            STATE["touches"] = []
            TOUCHES_PATH.unlink(missing_ok=True)
            BOARD_POSE_PATH.unlink(missing_ok=True)
            STATE["target"] = STATE["order"][0]
            STATE["anchor_rms"] = None
            STATE["guidance"] = ("Anchor cleared. Touch corners at the "
                                 "board's current placement and re-solve.")
        elif what == "static":
            STATE["acc"] = {c: {"m": [], "o3": [], "pix": [], "intr": None,
                                "heights": []} for c in STATIC_CAMS}
            STATE["install"] = None
    return jsonify(ok=True)


@app.route("/state")
def state():
    acc = {c: {"pairs": len(STATE["acc"][c]["m"]),
               "heights_mm": sorted({round(x * 1000) for x in STATE["acc"][c]["heights"]})}
           for c in STATIC_CAMS}
    return jsonify(target=STATE["target"],
                   corners_seen=STATE.get("corners_seen", {}),
                   touches=STATE["touches"], anchor_rms=STATE["anchor_rms"],
                   acc=acc, guidance=STATE["guidance"], install=STATE["install"],
                   anchored=BOARD_POSE_PATH.exists())


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<title>FR3 calibration</title><style>
body{font-family:system-ui;margin:0;background:#111;color:#eee;display:flex}
#side{width:380px;padding:14px;background:#1a1a1a;height:100vh;overflow:auto;flex:none}
#imgwrap{flex:1;height:100vh;overflow:auto;display:flex;flex-direction:column;gap:4px;padding:4px;align-items:center;justify-content:center}
.cambox{position:relative}
.cambox img{max-width:100%;max-height:45vh;width:auto;height:auto;display:block;cursor:crosshair;min-height:40px;background:#000}
.camlbl{position:absolute;top:6px;left:8px;font:600 12px monospace;color:#6f9;background:rgba(0,0,0,.55);padding:2px 7px;border-radius:5px;pointer-events:none}
h3{margin:6px 0}.sec{border:1px solid #333;border-radius:8px;padding:10px;margin:10px 0;background:#202020}
button{padding:8px 12px;margin:4px 2px;border:0;border-radius:6px;background:#2a6;color:#fff;cursor:pointer;font-size:13px}
button.alt{background:#444}button.go{background:#c33}
input{width:80px;padding:6px;border-radius:6px;border:1px solid #444;background:#111;color:#eee}
label{font-size:13px}#status{font:12px monospace;color:#fc6;margin:8px 0;min-height:30px;white-space:pre-wrap}
.row{font:12px monospace;color:#9cf;margin:3px 0}.ok{color:#6f9}.bad{color:#f88}
</style></head><body>
<div id=side>
<h3>FR3 calibration</h3>
<div><button onclick=refreshAll()>refresh photos</button>
<span style=font-size:11px;color:#888>photos update ONLY when you click
(or after Record/tap), no background streaming</span></div>
<div id=corners class=row></div>
<div id=status>loading...</div>

<div class=sec>
<h3>1. Anchor (touch corners)</h3>
<div style=font-size:12px;color:#aaa>Drive the gripper tip onto the red "GO HERE"
corner (tap either photo to pick a different one), then Record. Spread across the board.</div>
<button class=go onclick=record()>Record touch</button>
<button class=alt onclick="reset('anchor')">reset</button>
<div id=touches></div>
<button onclick=solveAnchor()>Solve + save anchor</button>
<div id=anchorRes class=row></div>
</div>

<div class=sec>
<h3>2. Heights (all cams)</h3>
<div style=font-size:12px;color:#aaa>Park the arm clear. Type the board height
(0 = flat on table; put it on a riser to get a few heights), then Capture.</div>
<div><label>height <input id=h type=number value=0 step=5> mm</label>
<button onclick=capHeight()>Capture all cams</button>
<button class=alt onclick="reset('static')">reset</button></div>
<div id=acc></div>
<button onclick=solveStatic()>Solve + install birdview & sideview</button>
<div id=staticRes class=row></div>
</div>
<div style=font-size:11px;color:#777>Wrist cam auto-drives the arm; run that from
the CLI (calibrate_wrist). After installing, restart the server to apply.</div>
</div>
<div id=imgwrap>
<div class=cambox><div class=camlbl>birdview</div><img id=img-birdview onclick="tap(event,'birdview')" alt="birdview loading..."></div>
<div class=cambox><div class=camlbl>sideview</div><img id=img-sideview onclick="tap(event,'sideview')" alt="sideview loading..."></div>
</div>
<script>
const CAMS=['birdview','sideview'];
function st(m){document.getElementById('status').textContent=m;}
// No background polling: frames load only on demand (refresh button, or a
// one-shot refresh after an action that moves the GO HERE marker).
function refreshAll(){CAMS.forEach(c=>{
 document.getElementById('img-'+c).src='/frame.jpg?v=4&cam='+c+'&t='+Date.now();});}
async function poll(){
 try{
  let s=await (await fetch('/state')).json();
  st(s.guidance);
  let cs=s.corners_seen||{};
  document.getElementById('corners').innerHTML=Object.keys(cs).map(c=>
   c+' sees <b>'+cs[c].n+'/'+cs[c].of+'</b> corners (as of '+cs[c].at+')').join(' &nbsp;|&nbsp; ')
   ||'corner counts appear after a photo refresh';
  let th='<div class=row>touches: '+s.touches.length+' [target corner '+(s.target==null?'-':s.target)+']</div>';
  s.touches.forEach(t=>{th+='<div class=row>corner '+t.corner_id+': ('+t.tcp_xyz.map(x=>x.toFixed(3))+')</div>';});
  document.getElementById('touches').innerHTML=th;
  if(s.anchor_rms!=null)document.getElementById('anchorRes').innerHTML=
    '<span class='+(s.anchor_rms<5?'ok':'bad')+'>anchor RMS '+s.anchor_rms.toFixed(1)+' mm</span>';
  let ah='';for(let c in s.acc){ah+='<div class=row>'+c+': '+s.acc[c].pairs+' pairs, heights '+
    (s.acc[c].heights_mm.join(',')||'-')+' mm</div>';}
  document.getElementById('acc').innerHTML=ah;
  if(s.install){let r='';for(let c in s.install){let v=s.install[c];
    r+=v.ok?('<div class=row ok>'+c+': RMSE '+v.rmse_mm+' mm, scale '+v.depth_scale+', offset '+v.depth_offset_mm+' mm</div>')
          :('<div class=row bad>'+c+': '+v.error+'</div>');}
    document.getElementById('staticRes').innerHTML=r;}
 }catch(e){st('app up, waiting for it... ('+e.message+')');}
}
async function tap(e,cam){let im=e.target,nw=im.naturalWidth;if(!nw)return;
 let r=im.getBoundingClientRect();
 let u=(e.clientX-r.left)*nw/r.width,v=(e.clientY-r.top)*im.naturalHeight/r.height;
 let res=await (await fetch('/tap',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({cam:cam,u:u,v:v})})).json();
 if(!res.ok)st(res.error);else{poll();refreshAll();}}
async function record(){let res=await (await fetch('/record_touch',{method:'POST'})).json();
 if(!res.ok){st('ERROR: '+res.error);return;}poll();refreshAll();}
async function solveAnchor(){let res=await (await fetch('/solve_anchor',{method:'POST'})).json();
 if(!res.ok){st('ERROR: '+res.error);return;}
 st('anchor RMS '+res.rms_mm.toFixed(1)+' mm, per-touch '+res.per_touch_mm.join(',')+' mm');poll();}
async function capHeight(){let h=document.getElementById('h').value;st('capturing all cams...');
 let res=await (await fetch('/capture_height',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({height_mm:h})})).json();
 if(!res.ok){st('ERROR: '+res.error);return;}poll();refreshAll();}
async function solveStatic(){st('solving...');
 let res=await (await fetch('/solve_static',{method:'POST'})).json();
 if(!res.ok){st('ERROR: '+res.error);return;}poll();}
async function reset(w){await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({what:w})});poll();}
// No background polling of any kind: state only changes through the
// buttons in this UI, and every button already calls poll() afterwards.
// Each photo load triggers one status poll so the corner counts above
// always describe the photos you are looking at.
CAMS.forEach(c=>document.getElementById('img-'+c).addEventListener('load',poll));
poll();
refreshAll();
</script></body></html>"""


if __name__ == "__main__":
    print(f"calib app on :8892  (server={sc.SERVER}, cals -> {CAL_DIR})")
    app.run(host="0.0.0.0", port=8892, threaded=True)
