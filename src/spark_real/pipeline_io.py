import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from spark_real.video_recorder import VideoRecorder
from spark_real.control.success_predicates import UNVERIFIED
from spark_real.pipeline_execution import CACHE_PLAN_SOURCES
from spark_real.pipeline_types import TaskResult
from spark_real.routes import streaming

logger = logging.getLogger(__name__)

# The message ScoreExecutor attaches when the operator hits stop. A robot
# reflex abort carries a different message and IS a real failure.
ABORT_MESSAGE = "Aborted by user"


def user_aborted(exec_results) -> bool:
    """
    True when the operator killed the run mid-execution.

    Accepts both the live ``ActionResult`` objects and the plain dicts the
    /scores page loads back from JSON, so one definition serves both.
    """
    for r in exec_results or []:
        message = r.get("message") if isinstance(r, dict) else getattr(r, "message", None)
        if ABORT_MESSAGE in (message or ""):
            return True
    return False


# Capture saving, video recording, and result persistence.
# Mixed into the pipeline class.
class IOMixin:

    def _save_captures(self, captures: Dict, timestamp: str) -> Dict[str, str]:
        """
        Save captured images to disk.
        """
        paths = {}
        out = Path(self.config.output_dir) / timestamp
        out.mkdir(parents=True, exist_ok=True)
        for cam_name, data in captures.items():
            rgb_path = str(out / f"{cam_name}_rgb.jpg")
            Image.fromarray(data["rgb"]).save(rgb_path)
            paths[f"{cam_name}_rgb"] = rgb_path
            if data["depth"] is not None:
                depth_path = str(out / f"{cam_name}_depth.npy")
                np.save(depth_path, data["depth"])
                paths[f"{cam_name}_depth"] = depth_path
        return paths

    def start_video_recording(self, label: str = "") -> None:
        """
        Begin server-side mp4 recording of every active camera, in
        parallel. One mp4 per camera (sideview, birdview, wrist where
        present), output:
            {output_dir}/videos/{ts}_{label}_{cam}.mp4
        Frames are pulled at ~10 fps directly from each Kinect /
        RealSense via the AzureKinectCamera.read() helper, so the
        recordings are independent of which camera is selected in the
        viewer and survive a closed browser.
        """
        if self._video_recorders:
            # Already recording, don't double-start.
            return

        out_dir = Path(self.config.output_dir).parent / "videos"

        def _make_provider(cam_name):
            def _grab():
                try:
                    rgb, _ = streaming.capture_single_camera(cam_name, need_depth=False)
                except Exception:
                    rgb = None
                return rgb

            return _grab

        # Figure out which cameras are actually live.
        live = []
        if self._kinect is not None:
            live.append("sideview")
        if self._kinect2 is not None:
            live.append("birdview")
        if self._realsense is not None:
            live.append("wrist")

        self._video_recorders = {}
        for cam in live:
            rec = VideoRecorder(_make_provider(cam), out_dir, fps=10)
            rec.start(f"{label}_{cam}" if label else cam)
            self._video_recorders[cam] = rec

    def stop_video_recording(self):
        """
        Stop every active recording and return a dict of {cam: path}.

        For backwards compat with callers that expected a single Path,
        returns just the first path if exactly one stream was recorded.
        """
        if not self._video_recorders:
            return None
        paths = {}
        for cam, rec in self._video_recorders.items():
            try:
                p = rec.stop()
                if p is not None:
                    paths[cam] = p
            except Exception as exc:
                logger.warning("stop_video_recording (%s) failed: %s", cam, exc)
        self._video_recorders = {}
        if len(paths) == 1:
            return next(iter(paths.values()))
        return paths or None

    def _maybe_save_bt(
        self,
        instruction,
        score,
        detections,
        success,
        save_to_library=None,
        executed=True,
        plan_source=None,
        bt_hash=None,
        user_aborted=False,
        verify_status=None,
    ):
        """
        Feed a finished run's outcome back into the BT library.

        This is the capture-once-reuse-forever hinge: a planner-on run that
        verifies successful lands here, gets stored with success=1, and every
        later run of the same task resolves to it with no LLM call.

        Two write paths, deliberately distinct:

          * plan came from the LLM  -> ``add()``: mint (or merge into) the
            entry for this (instruction, tree) pair.
          * plan was SERVED from the library -> ``bump()`` the entry that
            actually ran, and record this run's phrasing as an alias on it.
            ``add()`` here would mint a near-duplicate keyed on the new
            phrasing.

        ``executed=False`` (dry run, no robot, or a plan-only call) never
        writes: the caller's ``success`` defaults to True when there are no
        execution results, so a dry run would otherwise mint a
        cache-eligible entry that nothing ever verified.

        ``verify_status`` is the tri-state verdict. ``unverified`` bumps
        neither success nor fail -- it reinforces nothing and suppresses
        nothing. Treating it as a failure would cool the cache on the
        verifier's own blind spots; treating it as a success would reinforce
        an unverified tree. It does mint the row (success=0, below the
        ``min_success`` floor that makes an entry servable) so the abstain is
        countable on /scores instead of leaving no trace at all.
        """
        if self._bt_library is None:
            return
        # No-adaptation control: a frozen library must not learn from the run
        # it is scoring. See PipelineConfig.bt_frozen.
        from spark_real.pipeline_execution import _bt_frozen

        if _bt_frozen(self):
            logger.info(
                "BT library FROZEN: not recording '%s'; this run plans with "
                "the context trial 1 had", instruction,
            )
            return
        gate = save_to_library if save_to_library is not None else self.config.save_to_library
        if not gate:
            return
        if not executed:
            logger.info(
                "BT library: not recording %r (nothing executed; a dry run "
                "must not mint a cache-eligible entry)",
                instruction,
            )
            return
        if user_aborted:
            # The operator hit stop. That is not a task outcome: penalising the
            # tree would blame it for a decision the operator made, and
            # crediting it would bank a success nobody verified. The /scores
            # page already excludes these runs; the library must agree.
            logger.info(
                "BT library: not recording %r (aborted by user, not a task outcome)",
                instruction,
            )
            return
        if not score:
            return

        objects = []
        for d in detections or []:
            # detections may be ObjectDetection or already plain dicts
            lbl = getattr(d, "label", None) or (d.get("label") if isinstance(d, dict) else None)
            if lbl:
                objects.append(lbl)

        if verify_status == UNVERIFIED:
            # No verdict was reached. Count it -- MINTING the row if this was
            # an LLM-planned run with no hash yet -- so a systematically
            # abstaining verifier is visible on /scores instead of the cache
            # silently going cold. success stays 0, which is below lookup()'s
            # min_success floor, so the row is countable but not yet servable.
            entry = self._bt_library.note_unverified(
                bt_hash, instruction=instruction, score=score, objects=objects
            )
            logger.info(
                "BT library: not recording %r (verification %s -- neither "
                "reinforces nor suppresses)%s",
                instruction,
                UNVERIFIED,
                f"; entry {entry.hash} unverified={entry.unverified}" if entry else "",
            )
            return
        try:
            served = plan_source in CACHE_PLAN_SOURCES and bool(bt_hash)
            if served:
                entry = self._bt_library.bump(bt_hash, bool(success), alias=instruction)
                if entry is not None:
                    logger.info(
                        "BT library: %s %s (%r) -> success=%d fail=%d%s",
                        "reinforced" if success else "penalised",
                        entry.hash,
                        entry.instruction,
                        entry.success,
                        entry.fail,
                        " [promoted]" if entry.promoted else "",
                    )
                    return
                logger.warning(
                    "BT library: served hash %s vanished before feedback; " "falling back to add()",
                    bt_hash,
                )
            entry = self._bt_library.add(instruction, score, objects=objects, success=bool(success))
            logger.info(
                "BT library: stored %s (%r) from %s -> success=%d fail=%d%s",
                entry.hash,
                entry.instruction,
                plan_source or "unknown",
                entry.success,
                entry.fail,
                " [promoted]" if entry.promoted else "",
            )
        except Exception as e:
            logger.warning("BT library save failed: %s", e)

    def _save_result(self, result: TaskResult, timestamp: str):
        """
        Save task result to JSON.
        """
        out = Path(self.config.output_dir) / timestamp
        out.mkdir(parents=True, exist_ok=True)
        result_dict = {
            "instruction": result.instruction,
            "timestamp": result.timestamp,
            "detections": result.detections,
            "plan": result.plan,
            "execution_results": result.execution_results,
            "success": result.success,
            "duration": result.duration,
            "captures": result.captures,
            # Plan provenance. Without it a saved run cannot be attributed to
            # a cached tree vs. a fresh LLM call after the fact, which is the
            # whole point of running collection offline.
            "plan_source": getattr(result, "plan_source", None),
            "bt_hash": getattr(result, "bt_hash", None),
            "label_resolutions": getattr(result, "label_resolutions", None) or {},
            # RoboInter proposals and their verdicts, when the extension ran.
            # Absent from the JSON on an off run rather than written as [],
            # so a saved result is byte-identical to a pre-RoboInter one.
            **(
                {"robointer": list(getattr(result, "robointer", None) or [])}
                if getattr(result, "robointer", None)
                else {}
            ),
            # Tri-state verdict + the evidence behind it. `success` above is
            # exactly (verify_status == "pass"); /scores must be able to tell
            # an unverified run from a failed one.
            "verify_status": getattr(result, "verify_status", "unverified"),
            "verify": getattr(result, "verify", None),
        }
        with open(out / "result.json", "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
