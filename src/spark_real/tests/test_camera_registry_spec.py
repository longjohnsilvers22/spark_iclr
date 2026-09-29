"""
Pure-spec checks for the unified camera registry. No hardware: only
perception.camera_registry.spec(profile) is exercised, which reads
profile.cameras() and returns a role -> entry dict without opening any
device. Run from src/ with PYTHONPATH=.:
  python spark_real/perception/test_camera_registry_spec.py
"""

from spark_real.config import SparkConfig, load_profile
from spark_real.perception.camera_registry import spec


# Expected Kinect serials per single-arm family. These are canaries: a change
# here should be a deliberate config edit, not an accident. Keep in sync with
# configs/<family>_default.yaml (birdview 627 is the elevated unit on both rigs;
# sideview differs — FR3 host uses 189..., host/ur10e uses 626...).
_EXPECTED_KINECT_SERIALS = {
    "franka": {"birdview": "000000000000", "sideview": "000000000000"},
    "ur10e": {"birdview": "000000000000", "sideview": "000000000000"},
}


def _check_single_arm(family):
    profile = load_profile(SparkConfig(family=family))
    s = spec(profile)
    # The two external Kinects plus the wrist RealSense, keyed by role.
    assert set(s) == {"birdview", "sideview", "wrist"}, (family, set(s))
    assert s["birdview"]["type"] == "kinect", s["birdview"]
    assert s["sideview"]["type"] == "kinect", s["sideview"]
    assert s["wrist"]["type"] == "realsense", s["wrist"]
    for role, serial in _EXPECTED_KINECT_SERIALS[family].items():
        assert s[role]["serial"] == serial, (family, role, s[role])
    print(f"SINGLE-ARM SPEC OK ({family}): roles={sorted(s)}")


def test_franka_camera_spec():
    _check_single_arm("franka")


def test_ur10e_camera_spec():
    _check_single_arm("ur10e")


def main():
    _check_single_arm("franka")
    _check_single_arm("ur10e")

    bimanual = load_profile(SparkConfig(family="bimanual_franka"))
    s = spec(bimanual)
    assert set(s) == {"external", "wrist_left", "wrist_right"}, set(s)
    assert s["external"]["type"] == "zed_mini", s["external"]
    assert s["wrist_left"]["type"] == "realsense", s["wrist_left"]
    assert s["wrist_right"]["type"] == "realsense", s["wrist_right"]
    # Wrist cams carry their arm association and TCP->camera offset.
    assert s["wrist_left"]["arm"] == "left", s["wrist_left"]
    assert s["wrist_right"]["arm"] == "right", s["wrist_right"]
    print(f"BIMANUAL SPEC OK: roles={sorted(s)} " f"(external={s['external']['type']})")

    print("CAMERA REGISTRY SPEC OK: every family resolves a role-keyed roster")


if __name__ == "__main__":
    main()
