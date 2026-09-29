"""The release height must come from the container's PERCEIVED geometry.

Measured on the rig, run 20260804_150848, "pick up the stuffed animal and place
in the bowl" (bt_library/baf971e34606.json)::

    [4/5] move_to_keypoint {'keypoint_label': 'blue bowl', 'offset_z': 0.15}
    TCP z at release: -0.079     bowl centroid z: -0.229     table: -0.278

The transport arrived (commanded XY was 6.8 mm off). The jaws then opened 15 cm
above the bowl centroid, ~10 cm above its rim, and the plushie bounced out.

``offset_z: 0.15`` is a CONSTANT the planner guessed. It is wrong for a 5 cm
bowl and equally wrong for a 25 cm bin, because a constant cannot know either.
The container's rim and interior floor are both measurable from the depth
already sampled inside its SAM3 mask, and the held object's own extent is
measurable the same way -- so the release height can be computed instead.

The numbers below are that run's:

  * bowl centroid ``-0.2287`` and the OBB half-extents ``(6.9, 1.9) cm`` come
    from the verify trace (the same source test_release_arrival_gate.py uses);
  * the table sits at ``-0.278``; the recorded release TCP was ``-0.0787``.

The bowl's rim/interior split and the plushie's height are NOT in that trace --
the run predates the fields this file is about -- so they are reconstructed
from a synthetic depth image built to the measured bowl (see BOWL_* below), and
every test that depends on them says so.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from spark_real.control import release_height as rh
from spark_real.control.score_executor import ScoreExecutor
from spark_real.perception.mask_geometry import mask_height_profile

# --- the measured run ------------------------------------------------------

BOWL_XY = (-0.8325, -0.0443)
BOWL_CENTROID_Z = -0.2287  # place waypoint (-0.0787) minus the plan's offset_z
TABLE_Z = -0.278
MEASURED_RELEASE_Z = -0.0787
PLAN_OFFSET_Z = 0.15

# Reconstructed bowl (see the module docstring): a ~20 cm bowl whose rim reads
# at the perceived centroid -- `_cloud_median_world` reports the 95th
# percentile of a bimodal mask cloud, and a bowl mask IS bimodal (rim ring
# high, interior floor low) -- standing on the table with ~1 cm of wall.
BOWL_RIM_Z = BOWL_CENTROID_Z
BOWL_INTERIOR_Z = -0.268
BOWL_RADIUS_M = 0.10

# Reconstructed plushie: grasped at its perceived top, bottom on the table.
PLUSHIE_TOP_Z = -0.190
PLUSHIE_BOTTOM_Z = TABLE_Z
# What _approach_target actually commands: perceived z + GRIPPER_OPEN_Z_OFFSET.
PLUSHIE_GRASP_Z = PLUSHIE_TOP_Z + ScoreExecutor.GRIPPER_OPEN_Z_OFFSET
PLUSHIE_DROP_M = PLUSHIE_GRASP_Z - PLUSHIE_BOTTOM_Z  # 8.3 cm hanging below the TCP


def bowl_detection(profile: bool = True) -> dict:
    det = {
        "label": "blue bowl",
        "position_3d": [BOWL_XY[0], BOWL_XY[1], BOWL_CENTROID_Z],
        "confidence": 0.914,
        "world_major_axis_rad": 0.0,
        "aspect_ratio": 6.9 / 1.9,
        "obb_minor_m": 0.038,
    }
    if profile:
        det["rim_z_m"] = BOWL_RIM_Z
        det["interior_z_m"] = BOWL_INTERIOR_Z
        det["height_samples"] = 4200.0
    return det


def plushie_detection(profile: bool = True) -> dict:
    det = {
        "label": "stuffed animal",
        "position_3d": [-0.7942, 0.4138, PLUSHIE_TOP_Z],
        "confidence": 0.88,
    }
    if profile:
        det["rim_z_m"] = PLUSHIE_TOP_Z
        det["interior_z_m"] = PLUSHIE_BOTTOM_Z
        det["height_samples"] = 3100.0
    return det


# --- 1. the measured run, recomputed ---------------------------------------


def resolve(plan_z, container=None, held=None, **kw):
    """resolve_release_z with the executor's own bindings, so the numbers in
    this file are the numbers `_place_release_z` would produce."""
    return rh.resolve_release_z(
        plan_z=plan_z,
        container=bowl_detection() if container is None else container,
        held=plushie_detection() if held is None else held,
        table_z=ScoreExecutor.TABLE_Z_FLOOR,
        grasp_z_offset=ScoreExecutor.GRIPPER_OPEN_Z_OFFSET,
        **kw,
    )


def test_measured_run_releases_just_above_the_bowl_not_15cm_up():
    """The whole point: geometry, not the planner's constant."""
    out = resolve(BOWL_CENTROID_Z + PLAN_OFFSET_Z)
    assert out.source == "geometry"
    assert out.held_drop_m == pytest.approx(PLUSHIE_DROP_M)

    # The held object's bottom ends up a couple of cm off the bowl's floor...
    object_bottom = out.z - out.held_drop_m
    assert object_bottom == pytest.approx(BOWL_INTERIOR_Z + rh.FLOOR_TARGET_GAP_M, abs=1e-9)
    # ... i.e. 2 cm of fall, against the 10.6 cm the run actually had (the
    # jaws opened 14.9 cm above the bowl floor with 8.3 cm of plushie hanging
    # below them).
    assert object_bottom - BOWL_INTERIOR_Z < 0.03
    assert (MEASURED_RELEASE_Z - PLUSHIE_DROP_M) - BOWL_INTERIOR_Z > 0.10

    # ... while the JAWS stay clear above the rim; nothing is inserted.
    assert out.z > BOWL_RIM_Z + rh.MIN_TCP_ABOVE_RIM_M
    # ... and the whole thing is far below where the run let go.
    assert out.z < MEASURED_RELEASE_Z - 0.05


def test_the_planners_constant_is_what_produced_the_measured_height():
    """Guard on the reconstruction itself: plan-only reproduces the run."""
    out = resolve(
        BOWL_CENTROID_Z + PLAN_OFFSET_Z, container=bowl_detection(profile=False)
    )
    assert out.z == pytest.approx(MEASURED_RELEASE_Z, abs=1e-4)


# --- 2. fall back rather than invent ---------------------------------------


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param({}, id="no-profile-at-all"),
        pytest.param({"rim_z_m": BOWL_RIM_Z}, id="rim-only"),
        pytest.param({"interior_z_m": BOWL_INTERIOR_Z}, id="interior-only"),
        pytest.param(
            {"rim_z_m": BOWL_RIM_Z, "interior_z_m": BOWL_INTERIOR_Z, "height_samples": 4.0},
            id="too-few-depth-samples",
        ),
        pytest.param(
            {
                "rim_z_m": BOWL_INTERIOR_Z,
                "interior_z_m": BOWL_RIM_Z,
                "height_samples": 4200.0,
            },
            id="inverted-rim-below-interior",
        ),
        pytest.param(
            {"rim_z_m": float("nan"), "interior_z_m": BOWL_INTERIOR_Z, "height_samples": 4200.0},
            id="nan-rim",
        ),
        pytest.param(
            {"rim_z_m": 3.0, "interior_z_m": BOWL_INTERIOR_Z, "height_samples": 4200.0},
            id="rim-implausibly-high",
        ),
    ],
)
def test_a_container_without_usable_depth_falls_back_to_the_plan(broken):
    det = bowl_detection(profile=False)
    det.update(broken)
    plan_z = BOWL_CENTROID_Z + PLAN_OFFSET_Z
    out = resolve(plan_z, container=det)
    assert out.z == pytest.approx(plan_z, abs=1e-9)
    assert out.source == "plan-no-geometry"
    assert out.geometric_z is None


def test_a_held_object_without_a_profile_still_computes_but_conservatively():
    """Unknown held extent must not silently become zero: a zero drop would put
    the TCP itself where the object's bottom belongs."""
    out = resolve(
        BOWL_CENTROID_Z + PLAN_OFFSET_Z, held=plushie_detection(profile=False)
    )
    assert out.source == "geometry"
    assert out.held_drop_m == pytest.approx(rh.HELD_DROP_FALLBACK_M)
    assert "fallback" in out.detail


# --- 3. the hard bounds ----------------------------------------------------


SWEEP_TABLE_Z = -0.30


def _sweep():
    """Containers 0-24 cm deep standing on a table at SWEEP_TABLE_Z, holding
    objects that hang 0-20 cm below the TCP, against any plan Z at all."""
    rng = np.random.default_rng(20260804)
    for _ in range(2000):
        interior = float(rng.uniform(SWEEP_TABLE_Z, -0.15))
        rim = interior + float(rng.uniform(0.0, 0.24))
        drop = float(rng.uniform(0.0, 0.20))
        plan = float(rng.uniform(-0.40, 0.60))
        yield interior, rim, drop, plan


def _sweep_case(interior, rim, drop):
    cont = {
        "position_3d": [0.0, 0.0, rim],
        "rim_z_m": rim,
        "interior_z_m": interior,
        "height_samples": 500.0,
    }
    held = {
        "position_3d": [0.0, 0.0, interior + drop],
        "rim_z_m": interior + drop,
        "interior_z_m": interior,
        "height_samples": 500.0,
    }
    return cont, held


def test_the_release_never_puts_the_object_below_the_interior_floor():
    for interior, rim, drop, plan in _sweep():
        cont, held = _sweep_case(interior, rim, drop)
        out = rh.resolve_release_z(
            plan_z=plan, container=cont, held=held, table_z=SWEEP_TABLE_Z
        )
        assert out.source == "geometry"
        object_bottom = out.z - out.held_drop_m
        assert object_bottom >= interior + rh.MIN_FLOOR_CLEARANCE_M - 1e-9, (
            f"interior={interior} rim={rim} drop={drop} plan={plan} -> {out}"
        )


def test_the_jaws_never_descend_below_the_rim_plane():
    """The invariant this test is named for: the TCP never crosses the rim.

    It used to assert the stronger `rim + MIN_TCP_ABOVE_RIM_M`, which is a
    fixed 2 cm MARGIN on top of that invariant rather than the invariant
    itself. That margin makes a recess shallower than 2 cm unreachable by
    construction: measured 2026-08-19 on a 1.7 cm foam cutout, seating the
    tool needed a TCP 6.4 mm above the rim -- safely outside the container --
    and the margin held the release 13.6 mm higher still, so the tool was
    dropped from above every time.

    So the guard is now derived (release_height, PART 3): a container deeper
    than the margin keeps the full 2 cm, and a shallower one keeps only as
    much as still lets the held object reach its floor target, floored at
    zero. Both halves are asserted below -- the hard invariant everywhere,
    and the full margin wherever it was always in force.
    """
    for interior, rim, drop, plan in _sweep():
        cont, held = _sweep_case(interior, rim, drop)
        out = rh.resolve_release_z(
            plan_z=plan, container=cont, held=held, table_z=SWEEP_TABLE_Z
        )
        # HARD invariant, every container: never below the rim plane.
        assert out.z >= rim - 1e-9, (
            f"TCP {out.z:.4f} crossed the rim plane {rim:.4f}"
        )
        # Deep containers are untouched by the derived guard.
        if (rim - interior) >= rh.MIN_TCP_ABOVE_RIM_M:
            assert out.z >= rim + rh.MIN_TCP_ABOVE_RIM_M - 1e-9


def test_a_plan_that_would_drive_the_jaws_into_the_bowl_is_raised():
    """The clamp is not decoration: offset_z 0 aims the TCP AT the centroid."""
    out = resolve(BOWL_CENTROID_Z)  # offset_z: 0.0
    assert out.clamped is True
    assert out.z > BOWL_CENTROID_Z
    assert out.z >= BOWL_RIM_Z + rh.MIN_TCP_ABOVE_RIM_M


def test_the_interior_floor_can_never_read_below_the_table():
    """Depth punching through a translucent bowl reads the table, not the bowl."""
    det = bowl_detection()
    det["interior_z_m"] = TABLE_Z - 0.05  # 5 cm below a table it is standing on
    out = rh.resolve_release_z(
        plan_z=BOWL_CENTROID_Z + PLAN_OFFSET_Z,
        container=det,
        held=plushie_detection(),
        table_z=TABLE_Z,
        grasp_z_offset=ScoreExecutor.GRIPPER_OPEN_Z_OFFSET,
    )
    assert out.interior_z == pytest.approx(TABLE_Z)


# --- 4. precedence ---------------------------------------------------------


def test_an_explicit_strict_offset_is_honoured_verbatim():
    plan_z = BOWL_CENTROID_Z + PLAN_OFFSET_Z
    out = resolve(plan_z, strict=True)
    assert out.z == pytest.approx(plan_z, abs=1e-9)
    assert out.source == "plan-strict"


def test_a_plan_already_lower_than_the_geometry_wins():
    """Geometry only ever LOWERS a release; it never raises one the operator
    deliberately put closer to the container."""
    geom = resolve(10.0)
    low_plan = geom.z - 0.005  # still inside the safe band
    out = resolve(low_plan)
    assert out.z == pytest.approx(low_plan, abs=1e-9)
    assert out.source == "geometry"


# --- 5. perception: rim + interior out of a real depth image ---------------


def _straight_down_camera(height_m: float = 0.9):
    """Camera at +Z looking straight down; world_z = height - depth."""
    cam_pos = np.array([0.0, 0.0, height_m])
    cam_mat = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return cam_pos, cam_mat


def _bowl_depth_image(w=200, h=200, cam_h=0.9, rim_px=18):
    """Depth image of the reconstructed bowl: a rim annulus around a floor."""
    cam_pos, cam_mat = _straight_down_camera(cam_h)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0)
    outer = min(w, h) / 2.0 - 4
    mask = (r <= outer).astype(np.uint8)
    depth = np.zeros((h, w), dtype=np.float64)
    depth[mask > 0] = cam_h - BOWL_INTERIOR_Z
    depth[(r > outer - rim_px) & (r <= outer)] = cam_h - BOWL_RIM_Z
    return mask, depth, cam_pos, cam_mat


def test_mask_height_profile_recovers_the_rim_and_the_interior_floor():
    mask, depth, cam_pos, cam_mat = _bowl_depth_image()
    prof = mask_height_profile(
        mask, depth, cam_pos, cam_mat, fx=600.0, fy=600.0, cx_k=100.0, cy_k=100.0,
        use_opencv=True,
    )
    assert prof is not None
    assert prof["rim_z"] == pytest.approx(BOWL_RIM_Z, abs=2e-3)
    assert prof["interior_z"] == pytest.approx(BOWL_INTERIOR_Z, abs=2e-3)
    assert prof["n_valid"] > 1000


def test_mask_height_profile_returns_none_when_the_mask_has_no_depth():
    mask, depth, cam_pos, cam_mat = _bowl_depth_image()
    depth[:] = 0.0  # every pixel invalid
    prof = mask_height_profile(
        mask, depth, cam_pos, cam_mat, fx=600.0, fy=600.0, cx_k=100.0, cy_k=100.0,
        use_opencv=True,
    )
    assert prof is None


# --- 6. the executor uses it ------------------------------------------------


class FakeArm:
    GRIPPER_TYPE = "robotiq_2f85"
    robot_family = "ur10e"
    SUPPORTS_URSCRIPT = True

    def __init__(self, xyz=(-0.7942, 0.4138, 0.0572)):
        self.tcp = np.array([*xyz, 2.2214, -2.2214, 0.0], dtype=float)

    def get_tcp_pose(self):
        return self.tcp.copy()

    def get_observation(self):
        return {"tcp_pose": self.tcp.copy()}

    def get_gripper_position(self, publish=True):
        return 216.7

    def is_object_detected(self, publish=True):
        return True

    def _publish_gripper_state(self, force=False):
        return None


def _place_executor(params_extra=None, container_profile=True):
    arm = FakeArm()
    ex = ScoreExecutor(
        arm,
        detection_map={
            "blue bowl": bowl_detection(profile=container_profile),
            "stuffed animal": plushie_detection(),
        },
        velocity=0.2,
    )
    ex._holding = True
    ex._active_grasp_label = "stuffed animal"
    captured = {}

    def fake_transport(target, **kwargs):
        captured["target"] = np.asarray(target, dtype=float).copy()
        return True

    ex._transport_to = fake_transport
    params = {"keypoint_label": "blue bowl", "offset_z": PLAN_OFFSET_Z}
    params.update(params_extra or {})
    ex._move_to_keypoint(params, time.time())
    return captured["target"]


def test_the_place_transport_is_sent_to_the_computed_height():
    target = _place_executor()
    expected = resolve(BOWL_CENTROID_Z + PLAN_OFFSET_Z).z
    assert target[2] == pytest.approx(expected, abs=1e-9)
    assert target[2] < MEASURED_RELEASE_Z - 0.05
    # XY is untouched -- the transport was never the problem.
    assert target[0] == pytest.approx(BOWL_XY[0])
    assert target[1] == pytest.approx(BOWL_XY[1])


def test_the_place_transport_keeps_the_plan_when_the_node_says_strict():
    target = _place_executor({"strict_offset_z": True})
    assert target[2] == pytest.approx(MEASURED_RELEASE_Z, abs=1e-4)


def test_a_pick_is_untouched():
    """Not holding -> not a place -> the release-height logic never runs."""
    arm = FakeArm()
    ex = ScoreExecutor(
        arm, detection_map={"blue bowl": bowl_detection()}, velocity=0.2
    )
    ex._holding = False
    seen = {}
    ex._approach_target = lambda target, **kw: seen.update(
        target=np.asarray(target, dtype=float).copy()
    )
    ex._move_to_keypoint({"keypoint_label": "blue bowl", "offset_z": 0.0}, time.time())
    assert seen["target"][2] == pytest.approx(BOWL_CENTROID_Z)


# --- 5. the release give-back (port step 5, sim it-10 bug class) ------------
#
# The static grasp_z_offset assumed the TCP closed at perceived_top +
# GRIPPER_OPEN_Z_OFFSET; the real grasp descends further (grasp depth,
# learned per-label offsets, recovery Z bias). Gripping N cm lower means
# the object hangs N cm LESS below the TCP, so every drop was silently
# N cm high. Behind SPARK_RELEASE_GIVEBACK / place.release_giveback.


def _giveback_target(monkeypatch, extra_descent_m=None, enabled=True):
    if enabled:
        monkeypatch.setenv("SPARK_RELEASE_GIVEBACK", "1")
    else:
        monkeypatch.delenv("SPARK_RELEASE_GIVEBACK", raising=False)
    arm = FakeArm()
    ex = ScoreExecutor(
        arm,
        detection_map={
            "blue bowl": bowl_detection(),
            "stuffed animal": plushie_detection(),
        },
        velocity=0.2,
    )
    ex._holding = True
    ex._active_grasp_label = "stuffed animal"
    ex._grasp_perception_target_z = PLUSHIE_TOP_Z
    if extra_descent_m is None:
        ex._actual_grasp_tcp_z = None
    else:
        ex._actual_grasp_tcp_z = (
            PLUSHIE_TOP_Z + ScoreExecutor.GRIPPER_OPEN_Z_OFFSET - extra_descent_m
        )
    captured = {}

    def fake_transport(target, **kw):
        captured["target"] = np.asarray(target, dtype=float).copy()
        return True

    ex._transport_to = fake_transport
    ex._move_to_keypoint(
        {"keypoint_label": "blue bowl", "offset_z": PLAN_OFFSET_Z}, time.time()
    )
    return captured["target"][2]


def test_giveback_lowers_release_by_exactly_the_extra_descent(monkeypatch):
    """(learned offset -1cm) + (grasp depth 2cm) = 3cm deeper close ->
    the drop model shrinks by 3cm and the release comes down 3cm."""
    baseline = _giveback_target(monkeypatch, extra_descent_m=0.0)
    deeper = _giveback_target(monkeypatch, extra_descent_m=0.03)
    assert baseline - deeper == pytest.approx(0.03, abs=1e-9)


def test_giveback_never_crosses_the_rim_bound(monkeypatch):
    """No double-count against the rim/interior geometry: a big retry-bias
    descent (5cm) shrinks the drop, but the MIN_TCP_ABOVE_RIM bound still
    governs -- the jaws never chase the give-back below the rim plane."""
    z = _giveback_target(monkeypatch, extra_descent_m=0.05)
    assert z >= BOWL_RIM_Z + rh.MIN_TCP_ABOVE_RIM_M - 1e-9


def test_giveback_off_by_default(monkeypatch):
    with_flag = _giveback_target(monkeypatch, extra_descent_m=0.03)
    without_flag = _giveback_target(
        monkeypatch, extra_descent_m=0.03, enabled=False
    )
    legacy = resolve(BOWL_CENTROID_Z + PLAN_OFFSET_Z).z
    assert without_flag == pytest.approx(legacy, abs=1e-9)
    assert with_flag < without_flag


def test_giveback_without_recorded_close_falls_back_to_constant(monkeypatch):
    z = _giveback_target(monkeypatch, extra_descent_m=None)
    legacy = resolve(BOWL_CENTROID_Z + PLAN_OFFSET_Z).z
    assert z == pytest.approx(legacy, abs=1e-9)


def test_strict_plan_bypasses_the_giveback(monkeypatch):
    monkeypatch.setenv("SPARK_RELEASE_GIVEBACK", "1")
    target = _place_executor({"strict_offset_z": True})
    assert target[2] == pytest.approx(MEASURED_RELEASE_Z, abs=1e-4)
