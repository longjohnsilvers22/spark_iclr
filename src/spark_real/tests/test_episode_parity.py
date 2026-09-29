"""
Schema parity between a SPARK-recorded episode and a human teleop episode.

Synthesizes a full episode from a fake driver + fake cameras (no robot, no
hardware, no thread), then asserts it against a real episode from the human
corpus key-for-key and dtype-for-dtype. If the two ever diverge, a VLA trained
on SPARK data stops being comparable to one trained on human data, which is the
entire point of the exercise — so this is the test that guards the result.

The human reference root comes from ``SPARK_HUMAN_EPISODES`` (e.g.
``/data/teleop_episodes``); the human-comparison tests skip when it is unset,
so no absolute personal path is baked into the repo. That directory is opened
strictly read-only. Everything SPARK writes goes to a temp dir.

Run directly::

    SPARK_HUMAN_EPISODES=/data/teleop_episodes \\
        python -m spark_real.tests.test_episode_parity
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytest

from spark_real.control.command_latch import (
    SOURCE_SERVO,
    SOURCE_VELOCITY,
    CommandLatch,
    command_source,
)
from spark_real.recording.action_source import ActionSampler
from spark_real.recording.naming import iter_episode_dirs
from spark_real.recording.schema import (
    ACTION_SOURCE_FINITE_DIFF,
    ACTION_SOURCE_MEASURED,
    ACTION_SOURCE_SERVO,
    ACTION_SOURCE_VELOCITY,
    CAMERA_SLOTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_ACCEPTABLE_FPS,
    METADATA_KEYS,
    MIN_ACCEPTABLE_FPS,
    MODE_AUTONOMOUS,
    NPZ_CORE_KEYS,
    PROFILE_HUMAN_PARITY,
    PROFILE_NATIVE,
)
from spark_real.recording.settings import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_TRAIN_CAMERAS,
    RecordingSettings,
)
from spark_real.recording.vla_dataset import (
    ALL_CAMERAS,
    load_episode,
    validate_episode,
)
from spark_real.recording.vla_recorder import DemoRecorder

HUMAN_ROOT_ENV = "SPARK_HUMAN_EPISODES"
N_FRAMES = 40
RECORD_HZ = 15.0


# fakes


class FakeDriver:
    """
    Minimal stand-in for ``UR10eDriver`` with the surfaces the recorder reads.

    It owns a real :class:`CommandLatch`, so the action path under test is the
    production one and not a mock of it.
    """

    def __init__(self, commanded: bool = True):
        self._latch = CommandLatch()
        self._t = 0.0
        self.commanded = commanded

    # motion the "controller" issues
    def send_velocity(self, velocity, acceleration=0.5, duration=0.1):
        self._latch.latch_velocity([float(x) for x in velocity])

    def latch_command_velocity(self, velocity, source=None):
        self._latch.latch_velocity(velocity, source)

    def set_gripper_position(self, position, speed=None, force=None):
        self._latch.latch_gripper(position)

    def get_last_command(self):
        return self._latch.snapshot()

    # state the recorder reads
    def step(self, i: int) -> None:
        self._t = i / RECORD_HZ

    def get_observation(self) -> dict:
        t = self._t
        return {
            "tcp_pose": np.array([-0.7 + 0.01 * t, 0.1, 0.3, 2.22, -2.22, 0.0]),
            "joint_positions": np.full(6, 0.1 * t),
            "joint_velocities": np.full(6, 0.01),
            "tcp_velocity": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "gripper_position": 0.0,
        }

    def get_tcp_force(self) -> np.ndarray:
        return np.zeros(6)


class FakeCameras:
    """``pipeline.capture``-shaped source. ``slots`` picks which cams exist."""

    def __init__(self, cams: List[str], size=(1280, 720)):
        self.cams = cams
        self.size = size
        self.calls = 0

    def __call__(self) -> Dict[str, Dict]:
        self.calls += 1
        w, h = self.size
        out: Dict[str, Dict] = {}
        for i, cam in enumerate(self.cams):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, i % 3] = (self.calls * 3) % 256
            out[cam] = {"rgb": rgb, "depth": None, "calibration": None}
        return out


def _settings(data_dir: Path, **kw) -> RecordingSettings:
    s = RecordingSettings(
        data_dir=str(data_dir),
        record_hz=RECORD_HZ,
        robot_ip="0.0.0.0",
        emit_video=False,
        emit_plot=False,
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def record_episode(
    data_dir: Path,
    task: str = "put the knife in the tray",
    cams: Optional[List[str]] = None,
    n: int = N_FRAMES,
    commanded: bool = True,
    cmd_vel: Optional[List[float]] = None,
    **setting_overrides,
) -> Path:
    """
    Drive a recorder deterministically for ``n`` frames and save.

    ``step_once`` is called directly with a synthetic wall clock, so the
    episode is reproducible and the test does not spend ``n / 15`` seconds
    sleeping.
    """
    cams = cams if cams is not None else ["sideview", "birdview", "wrist"]
    robot = FakeDriver()
    capture = FakeCameras(cams)
    settings = _settings(data_dir, **setting_overrides)
    rec = DemoRecorder(settings, task=task, mode=MODE_AUTONOMOUS)
    rec.begin(capture_fn=capture, robot=robot, start_thread=False)

    t0 = time.time()
    for i in range(n):
        robot.step(i)
        if commanded:
            # A controller issuing a per-tick command, exactly as
            # CartesianServo does inside its PD loop.
            with command_source(SOURCE_SERVO):
                robot.latch_command_velocity(
                    list(cmd_vel) if cmd_vel is not None else [0.02, 0.0, -0.01, 0.0, 0.0, 0.0]
                )
            robot.set_gripper_position(1.0 if i > n // 2 else 0.0)
        rec.step_once(t0 + i / RECORD_HZ)
    rec.end(success=True, spark={"bt_hash": "deadbeef", "plan_source": "exact"})
    return rec.episode_dir


# fixtures


@pytest.fixture(scope="module")
def spark_episode():
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        yield record_episode(Path(tmp))


@pytest.fixture(scope="module")
def human_episode() -> Path:
    root = os.environ.get(HUMAN_ROOT_ENV)
    if not root:
        pytest.skip(f"{HUMAN_ROOT_ENV} unset; human-corpus comparison skipped")
    for ep in iter_episode_dirs(root):
        return ep
    pytest.skip(f"no episodes under {root}")


# structural self-checks (no human corpus needed)


def test_camera_role_map_is_the_rig_mapping():
    """birdview -> camera_1 and wrist -> wrist, established from real frames."""
    assert DEFAULT_CAMERA_MAP == {
        "sideview": "camera_0",
        "birdview": "camera_1",
        "wrist": "wrist",
    }
    assert DEFAULT_TRAIN_CAMERAS == ["camera_1", "wrist"]


def test_command_latch_mirrors_match_schema():
    """command_latch restates the source names; they must not drift."""
    assert SOURCE_VELOCITY == ACTION_SOURCE_VELOCITY
    assert SOURCE_SERVO == ACTION_SOURCE_SERVO


def test_layout(spark_episode: Path):
    assert (spark_episode / "trajectory.npz").is_file()
    assert (spark_episode / "metadata.json").is_file()
    meta = json.loads((spark_episode / "metadata.json").read_text())
    assert meta["cameras"] == ["camera_0", "camera_1", "wrist"]
    for slot in meta["cameras"]:
        d = spark_episode / "images" / slot
        assert d.is_dir(), f"missing {slot}"
        names = sorted(p.name for p in d.glob("frame_*.jpg"))
        assert names == [f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]


def test_images_are_640x480_jpeg(spark_episode: Path):
    img = cv2.imread(str(spark_episode / "images" / "camera_1" / "frame_0000.jpg"))
    assert img is not None
    assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def test_action_channel_matches_gripper(spark_episode: Path):
    with np.load(spark_episode / "trajectory.npz") as d:
        assert np.array_equal(d["actions"][:, 6], d["gripper_positions"])


def test_timestamps_and_rate(spark_episode: Path):
    meta = json.loads((spark_episode / "metadata.json").read_text())
    with np.load(spark_episode / "trajectory.npz") as d:
        ts = d["timestamps"]
    assert ts[0] == 0.0
    assert np.all(np.diff(ts) > 0)
    assert MIN_ACCEPTABLE_FPS <= meta["actual_fps"] <= MAX_ACCEPTABLE_FPS


def test_actions_are_commanded(spark_episode: Path):
    """The whole reason B1 existed: no NaN column, provenance is a command."""
    with np.load(spark_episode / "trajectory.npz") as d:
        assert np.all(np.isfinite(d["actions"]))
        sources = set(str(s) for s in d["action_source"])
    assert sources == {ACTION_SOURCE_SERVO}


# A command that actually exercises the rotation channels: SPARK yaws the wrist
# for an oriented grasp and rolls it for a tilt release, so wx/wz are genuinely
# nonzero on the autonomous path even though the human corpus has them at zero.
_ROTATING_CMD = [0.02, 0.0, -0.01, 0.05, 0.0, -0.03]


def test_human_parity_zeroes_wx_wz():
    """The opt-in ablation profile flattens the rotation channels."""
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        ep = record_episode(
            Path(tmp), n=8, cmd_vel=_ROTATING_CMD, action_profile=PROFILE_HUMAN_PARITY
        )
        with np.load(ep / "trajectory.npz") as d:
            assert np.all(d["actions"][:, 3] == 0.0)
            assert np.all(d["actions"][:, 5] == 0.0)


def test_native_is_the_default_and_preserves_commanded_rotation():
    """
    The default profile logs what the executor actually commanded.

    Zeroing wx/wz would contradict the frames recorded alongside them -- a label
    error, not a distribution shift. Parity with the human corpus is enforced at
    execution time (keep the motion top-down), never by rewriting the log.
    """
    assert RecordingSettings().action_profile == PROFILE_NATIVE
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        ep = record_episode(Path(tmp), n=8, cmd_vel=_ROTATING_CMD)
        with np.load(ep / "trajectory.npz") as d:
            assert np.any(d["actions"][:, 3] != 0.0)
            assert np.any(d["actions"][:, 5] != 0.0)


def test_missing_sideview_is_black_filled_not_dropped():
    """
    Default `fill` policy: an absent required slot is substituted, never silent.

    The training converter SKIPS an episode missing a camera directory, so a
    bird+wrist session would vanish at conversion time. Black-filling keeps it
    ingestible; metadata records the substitution so it stays auditable.
    """
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        ep = record_episode(Path(tmp), cams=["birdview", "wrist"], n=8)
        meta = json.loads((ep / "metadata.json").read_text())
        assert meta["cameras"] == ["camera_0", "camera_1", "wrist"]
        assert meta["spark"]["synthetic_cameras"] == ["camera_0"]
        filled = sorted((ep / "images" / "camera_0").glob("frame_*.jpg"))
        assert len(filled) == 8
        assert validate_episode(load_episode(ep, cameras=ALL_CAMERAS)) == []


def test_episode_valid_without_sideview_when_not_required():
    """Narrowing required_cameras is the way to genuinely omit a slot."""
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        ep = record_episode(
            Path(tmp),
            cams=["birdview", "wrist"],
            n=8,
            required_cameras=["camera_1", "wrist"],
        )
        meta = json.loads((ep / "metadata.json").read_text())
        assert meta["cameras"] == ["camera_1", "wrist"]
        assert not (ep / "images" / "camera_0").exists()
        assert validate_episode(load_episode(ep, cameras=ALL_CAMERAS)) == []


def test_late_camera_is_backfilled():
    """A camera that comes up mid-episode must not shift its own indices."""
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        robot = FakeDriver()
        capture = FakeCameras(["birdview"])
        settings = _settings(Path(tmp), max_black_frames=99)
        rec = DemoRecorder(settings, task="late cam", mode=MODE_AUTONOMOUS)
        rec.begin(capture_fn=capture, robot=robot, start_thread=False)
        t0 = time.time()
        for i in range(10):
            if i == 4:
                capture.cams = ["birdview", "wrist"]
            rec.step_once(t0 + i / RECORD_HZ)
        rec.end(success=True)
        ep = rec.episode_dir
        assert rec.black_frames == 4
        for slot in ("camera_1", "wrist"):
            names = sorted(p.name for p in (ep / "images" / slot).glob("*.jpg"))
            assert names == [f"frame_{i:04d}.jpg" for i in range(10)], slot
        assert validate_episode(load_episode(ep, cameras=ALL_CAMERAS)) == []


def test_episode_numbering_increments():
    with tempfile.TemporaryDirectory(prefix="spark_parity_") as tmp:
        a = record_episode(Path(tmp), n=4)
        b = record_episode(Path(tmp), n=4)
        assert a.name == "episode_0000"
        assert b.name == "episode_0001"


def test_action_fallbacks():
    """No latch -> measured twist; no twist either -> finite difference."""
    settings = _settings(Path("/nonexistent"))
    sampler = ActionSampler(settings, robot=None)
    proprio = {
        "tcp_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tcp_velocity": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    _, src = sampler.sample(proprio)
    assert src == ACTION_SOURCE_MEASURED

    sampler2 = ActionSampler(settings, robot=None)
    sampler2.sample({"tcp_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    action, src2 = sampler2.sample({"tcp_pose": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]})
    assert src2 == ACTION_SOURCE_FINITE_DIFF
    # 0.1 m in one 15 Hz tick is 1.5 m/s, clipped to the 0.2 m/s cap.
    assert action[0] == pytest.approx(settings.linear_velocity_clip * 0.4)


def test_stale_latch_is_not_used():
    settings = _settings(Path("/nonexistent"))
    robot = FakeDriver()
    robot.send_velocity([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    sampler = ActionSampler(settings, robot)
    sampler.max_latch_age = -1.0  # force every latch to read as stale
    _, src = sampler.sample({"tcp_velocity": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]})
    assert src == ACTION_SOURCE_MEASURED


# parity against the real human corpus


def test_npz_keys_match_human(spark_episode: Path, human_episode: Path):
    with (
        np.load(human_episode / "trajectory.npz") as h,
        np.load(spark_episode / "trajectory.npz") as s,
    ):
        human_keys = set(h.files)
        spark_keys = set(s.files)
        assert human_keys == set(NPZ_CORE_KEYS), (
            "the human corpus no longer matches schema.NPZ_CORE_KEYS; "
            f"human={sorted(human_keys)}"
        )
        assert human_keys <= spark_keys, f"SPARK is missing {human_keys - spark_keys}"
        for key in NPZ_CORE_KEYS:
            assert h[key].dtype == s[key].dtype, key
            assert h[key].ndim == s[key].ndim, key
            if h[key].ndim == 2:
                assert h[key].shape[1] == s[key].shape[1], key


def test_metadata_keys_match_human(spark_episode: Path, human_episode: Path):
    human = json.loads((human_episode / "metadata.json").read_text())
    spark = json.loads((spark_episode / "metadata.json").read_text())
    assert list(human.keys()) == list(METADATA_KEYS)
    # The human keys come first, in the human order; SPARK's extras follow.
    assert list(spark.keys())[: len(METADATA_KEYS)] == list(METADATA_KEYS)
    for key in METADATA_KEYS:
        assert type(spark[key]) is type(human[key]), key
    assert spark["task"] == spark["prompt"] == Path(spark_episode).parent.name


def test_human_frame_naming_matches(spark_episode: Path, human_episode: Path):
    human_meta = json.loads((human_episode / "metadata.json").read_text())
    for slot in human_meta["cameras"]:
        assert slot in CAMERA_SLOTS, f"unexpected human camera slot {slot!r}"
        h = sorted((human_episode / "images" / slot).glob("frame_*.jpg"))
        assert h[0].name == "frame_0000.jpg"
        assert len(h) == human_meta["num_frames"]


def test_one_loader_reads_both(spark_episode: Path, human_episode: Path):
    """The comparison is only valid if both corpora go through this path."""
    human = load_episode(human_episode)
    spark = load_episode(spark_episode)
    assert validate_episode(human) == []
    assert validate_episode(spark) == []
    assert not human.is_spark
    assert spark.is_spark
    # Amendment B: the training subset is bird + wrist on both sides.
    assert human.cameras == DEFAULT_TRAIN_CAMERAS
    assert spark.cameras == DEFAULT_TRAIN_CAMERAS
    assert load_episode(human_episode, cameras=ALL_CAMERAS).cameras == list(human.available_cameras)
    for ep in (human, spark):
        obs = ep.observations(0)
        assert set(obs) == set(DEFAULT_TRAIN_CAMERAS)
        for img in obs.values():
            assert img.ndim == 3 and img.shape[2] == 3


def test_human_action_manifold(human_episode: Path):
    """Documents what human_parity is imitating; guards the assumption."""
    with np.load(human_episode / "trajectory.npz") as h:
        actions = h["actions"]
        assert np.all(actions[:, 3] == 0.0)
        assert np.all(actions[:, 5] == 0.0)
        assert np.array_equal(actions[:, 6], h["gripper_positions"])


def _main() -> int:
    return pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    raise SystemExit(_main())
