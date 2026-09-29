"""
Unit test for the closed-loop scene-progress evaluator.

Proves the generic re-detection logic in ExecutionMixin classifies targets
correctly without any hardware: destinations are inferred from the score
across primitive shapes (pick-place, place_in_slot, sweep), and a target is
"handled" when it sits inside a destination zone or its base label is in the
executor's placed set. Everything else is unhandled work the outer loop
keeps acting on.

Run: python -m spark_real.tests.test_closed_loop_progress   (exit 0 = pass)
Also collectible by pytest (test_* functions, plain asserts).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spark_real.pipeline_execution import ExecutionMixin


def _det(label, xyz, minor=0.0, conf=0.9):
    return SimpleNamespace(
        label=label, position_3d=list(xyz), confidence=conf, obb_minor_m=minor
    )


class _StubPipeline(ExecutionMixin):
    # Minimal stand-in: only the pieces evaluate_scene_progress touches.
    def __init__(self, dets):
        self._dets = dets
        self.config = SimpleNamespace(closed_loop_clear_radius_m=0.12)

    def capture(self):
        return {"birdview": {"rgb": object()}}

    def detect(self, captures, prompts, multi_instance=False):
        return list(self._dets)

    def merge_detections(self, dets):
        return list(dets)


def test_destinations_inferred_across_primitives():
    # move_to_keypoint after a grasp -> destination
    pick_place = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
                {"type": "grasp", "params": {}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
                {"type": "release", "params": {}},
            ],
        }
    }
    d = ExecutionMixin._closed_loop_destinations(_StubPipeline([]), pick_place)
    assert "tray" in d and "fork 1" not in d, d

    # place_in_slot container_label and sweep target_label
    mixed = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "place_in_slot", "params": {"container_label": "dish rack"}},
                {
                    "type": "sweep",
                    "params": {"target_label": "dustpan", "area_label": "crumb"},
                },
            ],
        }
    }
    d2 = ExecutionMixin._closed_loop_destinations(_StubPipeline([]), mixed)
    assert "dish rack" in d2 and "dustpan" in d2, d2


def test_handled_vs_unhandled():
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
                {"type": "grasp", "params": {}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
                {"type": "release", "params": {}},
            ],
        }
    }
    dets = [
        _det("tray", (0.60, 0.00, -0.10), minor=0.30),  # destination
        _det("fork 1", (0.61, 0.02, -0.05)),  # inside tray zone
        _det("fork 2", (0.20, 0.30, -0.05)),  # out on the table
        _det("knife 1", (0.25, -0.20, -0.05)),  # out, but placed
    ]
    p = _StubPipeline(dets)
    out = p.evaluate_scene_progress(
        score,
        prompts=["fork", "knife", "tray"],
        handled_labels={"knife 1"},
        captures={"birdview": {"rgb": object()}},
    )

    assert "tray" not in out["unhandled"], out  # receptacle excluded
    assert "fork 1" in out["handled"], out  # in destination zone
    assert "knife 1" in out["handled"], out  # placed-label set
    assert out["unhandled"] == ["fork 2"], out  # only real remainder


def test_goal_holds_when_all_in():
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
                {"type": "grasp", "params": {}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
                {"type": "release", "params": {}},
            ],
        }
    }
    dets = [
        _det("tray", (0.60, 0.0, -0.10), minor=0.30),
        _det("fork 1", (0.60, 0.01, -0.05)),  # landed inside the tray zone
    ]
    out = _StubPipeline(dets).evaluate_scene_progress(
        score,
        prompts=["fork", "tray"],
        handled_labels=set(),
        captures={"birdview": {"rgb": object()}},
    )
    assert out["unhandled"] == [], out


def main():
    test_destinations_inferred_across_primitives()
    test_handled_vs_unhandled()
    test_goal_holds_when_all_in()
    print("closed-loop progress evaluator: all checks passed")


if __name__ == "__main__":
    main()
