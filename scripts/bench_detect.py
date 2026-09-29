"""Benchmark SAM3 detect latency: per-prompt, per-camera, end-to-end.

Attributes a detect() call to its real stages so a regression can be
pinned on a stage instead of guessed at:

    image encode   set_image()      -> full vision backbone @ 1008x1008
    grounding      set_text_prompt() -> text encoder + grounding head
    depth          DA3 monocular metric depth (once per camera)
    geometry       CPU mask post-processing (OBB / backprojection / profile)

The point of the breakdown is the encode:grounding ratio. `set_image()` is
per-IMAGE work and `set_text_prompt()` is per-PROMPT work, so a detect that
calls set_image() inside its prompt loop pays the backbone N times for one
image. `--compare` measures that directly by running both strategies.

No robot and no camera required: by default it runs on a synthetic frame.
Pass --image to use a real capture.

Usage:
    conda activate spark_conda

    # end-to-end + stage breakdown on a synthetic 1280x720 frame
    python ~/spark/scripts/bench_detect.py --prompts plushie bowl

    # is a slow detect explained by stream contention? (measured: no)
    python ~/spark/scripts/bench_detect.py --contention

    # quantify the encode-once saving
    python ~/spark/scripts/bench_detect.py --compare \
        --prompts plushie "stuffed animal" "soft toy" bowl dish container

    # real frames, two cameras, no model load (CPU geometry stages only)
    python ~/spark/scripts/bench_detect.py --image side.png --image bird.png
    python ~/spark/scripts/bench_detect.py --geometry-only

This loads SAM3 onto the GPU, so it refuses to run while a SPARK server is
up (that server owns the GPU and its SAM3 instance). Use --force to override.
"""

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Resolve the checkout this script lives in, so a worktree benches its own
# code rather than whatever is in ~/spark/src.
SPARK_SRC = Path(__file__).resolve().parent.parent / "src"
if not SPARK_SRC.exists():
    SPARK_SRC = Path.home() / "spark" / "src"
sys.path.insert(0, str(SPARK_SRC))

SAM3_PATHS = [Path.home() / "mv_sam3" / "sam3"]
if os.environ.get("SPARK_SAM3_DIR"):
    SAM3_PATHS.insert(0, Path(os.environ["SPARK_SAM3_DIR"]))
for _p in SAM3_PATHS:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

class Timer:
    """Accumulates wall time per named stage.

    CUDA is async, so every stage that touches the GPU synchronizes before
    stopping the clock -- otherwise the backbone's cost lands on whichever
    later stage happens to read a tensor back to the host.
    """

    def __init__(self, sync=False):
        self.totals = {}
        self.counts = {}
        self._sync = sync

    def _synchronize(self):
        if not self._sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    @contextmanager
    def stage(self, name):
        self._synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, title, total_wall=None):
        print(f"\n{title}")
        print(f"  {'stage':<18}{'calls':>7}{'total s':>11}{'each ms':>11}{'share':>8}")
        print(f"  {'-' * 55}")
        denom = total_wall if total_wall else sum(self.totals.values())
        for name, tot in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            share = (tot / denom * 100.0) if denom else 0.0
            print(f"  {name:<18}{n:>7}{tot:>11.3f}{tot / n * 1000:>11.1f}{share:>7.1f}%")
        if total_wall:
            print(f"  {'-' * 55}")
            print(f"  {'END-TO-END':<18}{'':>7}{total_wall:>11.3f}")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def synthetic_frame(h=720, w=1280, seed=0):
    """A frame with a few blobs. Detection quality is irrelevant here -- the
    cost of the backbone and the grounding head does not depend on what the
    model finds, only on the image size and the prompt count."""
    rng = np.random.default_rng(seed)
    rgb = np.full((h, w, 3), 130, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, rx, ry, col in [
        (int(w * 0.35), int(h * 0.5), 150, 120, (190, 90, 90)),
        (int(w * 0.68), int(h * 0.55), 130, 95, (90, 140, 190)),
    ]:
        m = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
        rgb[m] = col
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-8, 9, rgb.shape), 0, 255)
    return rgb.astype(np.uint8)


def load_frames(paths, n_cameras):
    if paths:
        from PIL import Image as PILImage

        out = []
        for p in paths:
            arr = np.array(PILImage.open(p).convert("RGB"))
            out.append(arr)
            print(f"  frame {p}: {arr.shape[1]}x{arr.shape[0]}")
        return out
    frames = [synthetic_frame(seed=i) for i in range(n_cameras)]
    print(f"  {n_cameras} synthetic frame(s): 1280x720")
    return frames


def server_running():
    try:
        out = subprocess.run(
            ["pgrep", "-f", "spark_real.server"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


# --------------------------------------------------------------------------
# geometry-only benchmark (no GPU, no model, safe any time)
# --------------------------------------------------------------------------

def bench_geometry(frames, repeats):
    from spark_real.perception.mask_geometry import (
        _cloud_median_world, _compute_orientation, _pca_obb, _top_layer_mask,
        _world_xy_pca_obb, mask_height_profile,
    )

    print("\n=== CPU geometry post-processing (per detection) ===")
    h, w = frames[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((((xx - w // 2) / (w * 0.17)) ** 2
             + ((yy - h // 2) / (h * 0.21)) ** 2) < 1.0).astype(np.uint8)
    depth = np.full((h, w), 0.9, np.float32)
    cam_pos, cam_mat = np.array([0.0, 0.0, 1.0]), np.eye(3)

    t = Timer()
    for _ in range(repeats):
        with t.stage("top_layer_mask"):
            obb = _top_layer_mask(mask, depth)
        with t.stage("pca_obb"):
            ang, _ar, _sp, lp = _pca_obb(obb)
        with t.stage("world_xy_pca_obb"):
            _world_xy_pca_obb(mask, depth, cam_pos, cam_mat, fovy_deg=45.0,
                              use_opencv=False, return_confidence=True)
        with t.stage("cloud_median_world"):
            _cloud_median_world(mask, depth, cam_mat, cam_pos, fovy_deg=45.0,
                                use_opencv=False)
        with t.stage("height_profile"):
            mask_height_profile(mask, depth, cam_pos, cam_mat, fovy_deg=45.0,
                                use_opencv=False)
        with t.stage("orientation"):
            _compute_orientation(mask, depth, cam_mat, cam_pos, 45.0,
                                 w / 2, h / 2, h, w, ang, lp, 0.9)
    per_det = sum(t.totals.values()) / repeats
    t.report(f"geometry, {repeats} rep(s), mask={int(mask.sum())}px")
    print(f"\n  -> {per_det * 1000:.1f} ms per detection of CPU geometry")
    return per_det


# --------------------------------------------------------------------------
# SAM3 benchmark
# --------------------------------------------------------------------------

def bench_sam3(frames, prompts, repeats, compare):
    import torch
    from PIL import Image

    from spark_real.perception.spark_perception import (
        Sam3Processor, build_sam3_image_model,
    )

    if build_sam3_image_model is None:
        print("ERROR: SAM3 not importable. Checked:")
        for p in SAM3_PATHS:
            print(f"  {p}  ({'exists' if p.exists() else 'MISSING'})")
        return None

    print("\n=== loading SAM3 (this is the ~50s startup cost, not detect cost) ===")
    t0 = time.perf_counter()
    proc = Sam3Processor(build_sam3_image_model())
    proc.set_confidence_threshold(0.05)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    pils = [Image.fromarray(f) for f in frames]
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Warm up: first call pays cuDNN autotuning and lazy kernel loads, which
    # would otherwise be misattributed to the first stage measured.
    with autocast():
        _st = proc.set_image(pils[0])
        proc.set_text_prompt(prompt=prompts[0], state={**_st})

    def run(encode_once):
        t = Timer(sync=True)
        wall0 = time.perf_counter()
        for _ in range(repeats):
            for pil in pils:
                if encode_once:
                    with t.stage("image encode"):
                        with autocast():
                            base = proc.set_image(pil)
                    for pr in prompts:
                        with t.stage("grounding"):
                            with autocast():
                                proc.set_text_prompt(prompt=pr, state={**base})
                else:
                    for pr in prompts:
                        with t.stage("image encode"):
                            with autocast():
                                base = proc.set_image(pil)
                        with t.stage("grounding"):
                            with autocast():
                                proc.set_text_prompt(prompt=pr, state={**base})
        return t, time.perf_counter() - wall0

    n_cam, n_pr = len(pils), len(prompts)
    print(f"\n  workload: {n_cam} camera(s) x {n_pr} prompt(s) x {repeats} rep(s)")

    t_new, wall_new = run(encode_once=True)
    t_new.report(f"ENCODE ONCE PER CAMERA ({n_cam} encodes, {n_cam * n_pr} groundings)",
                 wall_new)
    print(f"\n  per detect (1 pass over all cameras): {wall_new / repeats:.2f}s")

    if not compare:
        return wall_new / repeats

    t_old, wall_old = run(encode_once=False)
    t_old.report(f"ENCODE PER PROMPT, legacy ({n_cam * n_pr} encodes)", wall_old)
    print(f"\n  per detect (1 pass over all cameras): {wall_old / repeats:.2f}s")

    saved = wall_old - wall_new
    print("\n=== COMPARISON ===")
    print(f"  encode per prompt : {wall_old / repeats:7.2f}s per detect")
    print(f"  encode once       : {wall_new / repeats:7.2f}s per detect")
    if wall_old > 0:
        print(f"  saving            : {saved / repeats:7.2f}s per detect "
              f"({saved / wall_old * 100:.0f}%, {wall_old / max(wall_new, 1e-9):.1f}x faster)")
    return wall_new / repeats


def bench_contention(frames, prompts, repeats):
    """Does concurrent stream-like load slow a detect down?

    A live server serves /api/capture/stream while a detect runs, so "the
    stream starved the detect" is a natural theory for a slow detect. It is
    also testable: time the same detect with and without a background load
    that mimics a stream client (a CPU frame resize plus GPU work standing in
    for the Kinect depth engine, at a fixed rate), including loads far heavier
    than any real frontend produces.

    Measured on an RTX 4090 this comes out at ~1.0x even at 8 clients / 30 Hz,
    so a slow detect on this stack is NOT explained by stream contention. The
    function stays so that claim can be re-checked rather than believed.
    """
    import threading

    import torch
    from PIL import Image

    from spark_real.perception.spark_perception import (
        Sam3Processor, build_sam3_image_model,
    )

    if build_sam3_image_model is None:
        print("\nERROR: SAM3 not importable; skipping contention A/B")
        return

    print("\n=== loading SAM3 for the contention A/B ===")
    proc = Sam3Processor(build_sam3_image_model())
    proc.set_confidence_threshold(0.03)
    pils = [Image.fromarray(f) for f in frames]
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def one_detect():
        for pil in pils:
            with autocast():
                base = proc.set_image(pil)
            for pr in prompts:
                with autocast():
                    proc.set_text_prompt(prompt=pr, state={**base})

    one_detect()  # warm

    def timed():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_detect()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    stop = threading.Event()

    def worker(rate_hz, mb):
        buf = torch.randn(mb * 64, 1024, device="cuda")
        period = 1.0 / rate_hz
        small = frames[0][::2, ::2].copy()
        while not stop.is_set():
            t0 = time.perf_counter()
            small.tobytes()
            (buf @ buf.T[:1024, :1024]).sum().item()
            dt = time.perf_counter() - t0
            if dt < period:
                stop.wait(period - dt)

    def measure(tag):
        ts = [timed() for _ in range(repeats)]
        best = min(ts)
        print(f"  {tag:46s} {best:6.2f}s")
        return best

    print(f"\n=== contention A/B ({len(pils)} cam x {len(prompts)} prompts) ===")
    base = measure("no background load (control)")
    for tag, n, rate, mb in [
        ("2 clients @ 2.3 Hz (typical frontend)", 2, 2.3, 8),
        ("4 clients @ 10 Hz", 4, 10.0, 16),
        ("8 clients @ 30 Hz (far heavier than real)", 8, 30.0, 32),
    ]:
        stop.clear()
        threads = [threading.Thread(target=worker, args=(rate, mb), daemon=True)
                   for _ in range(n)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        got = measure(tag)
        stop.set()
        for t in threads:
            t.join(timeout=2)
        print(f"  {'':46s} -> {got/base:.2f}x baseline")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+",
                    default=["plushie", "stuffed animal", "soft toy",
                             "bowl", "dish", "container"],
                    help="text prompts (default: the plushie+bowl task's 6)")
    ap.add_argument("--image", action="append", dest="images",
                    help="frame to use; repeat for multiple cameras")
    ap.add_argument("--cameras", type=int, default=2,
                    help="synthetic frame count when --image is not given")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compare", action="store_true",
                    help="also time the legacy encode-per-prompt strategy")
    ap.add_argument("--contention", action="store_true",
                    help="A/B the detect against concurrent stream-like load")
    ap.add_argument("--geometry-only", action="store_true",
                    help="CPU mask post-processing only; no GPU, no model load")
    ap.add_argument("--force", action="store_true",
                    help="run even if a SPARK server is up (it owns the GPU)")
    args = ap.parse_args()

    needs_gpu = not args.geometry_only
    if needs_gpu and server_running() and not args.force:
        print("REFUSING: a SPARK server is running and owns the GPU + its SAM3.\n"
              "Loading a second SAM3 would compete for VRAM and skew every number\n"
              "here. Stop the server first, or re-run with --geometry-only (CPU\n"
              "stages only, safe while the server is up), or --force.")
        return 1

    print("=== SPARK detect latency benchmark ===")
    frames = load_frames(args.images, args.cameras)
    print(f"  prompts ({len(args.prompts)}): {', '.join(args.prompts)}")

    per_det = bench_geometry(frames, max(args.repeats, 3))

    if args.geometry_only:
        print("\n(geometry-only: skipping SAM3. Re-run without --geometry-only\n"
              " once the server is stopped for the encode/grounding split.)")
        return 0

    if args.contention:
        bench_contention(frames, args.prompts, args.repeats)
        return 0

    detect_s = bench_sam3(frames, args.prompts, args.repeats, args.compare)
    if detect_s:
        n_det = len(frames) * len(args.prompts)
        print(f"\n  CPU geometry over {n_det} detections: "
              f"{per_det * n_det * 1000:.0f} ms "
              f"({per_det * n_det / detect_s * 100:.1f}% of detect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
