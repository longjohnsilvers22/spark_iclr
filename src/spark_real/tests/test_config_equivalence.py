# Equivalence oracle for the config spine.
#
# Proves spark_real.config.load_profile(...).to_pipeline_kwargs() reproduces
# the historical inline config-building logic in server.py for every robot
# family and flag combination, and that the result constructs a real
# PipelineConfig. Run before and after wiring config.py into server.py.

import yaml

from spark_real.config import SparkConfig, load_profile, CONFIGS_DIR

# Copies of the per-family fallbacks from the historical server.py path.
_FALLBACK_MODELS = {
    "ur10e": "UR10e",
    "franka": "Franka FR3",
    "g1": "Unitree G1",
    "bimanual_franka": "Bimanual Franka (Panda+FR3)",
}
_FALLBACK_IPS = {
    "ur10e": "192.168.56.101",
    "franka": "172.16.0.2",
    "g1": "192.168.123.161",
    "bimanual_franka": "172.16.0.102",
}


def golden_kwargs(family, ip=None, no_robot=False, no_kinect=False, strict=False):
    # Faithful transcription of server.py main()'s config-building block.
    model_str = _FALLBACK_MODELS.get(family, family)
    ip_str = _FALLBACK_IPS.get(family, "192.168.56.101")
    use_realsense_cfg = None
    use_hw_sync_cfg = None
    depth = res = fps = None
    try:
        cfg_path = CONFIGS_DIR / f"{family}_default.yaml"
        if cfg_path.exists():
            with open(cfg_path) as handle:
                cfg = yaml.safe_load(handle) or {}
            robot_section = cfg.get("robot", {}) or {}
            variant = robot_section.get("variant")
            if variant:
                model_str = str(variant)
            yaml_ip = robot_section.get("ip")
            if yaml_ip:
                ip_str = str(yaml_ip)
            if "use_realsense" in cfg and cfg["use_realsense"] is not None:
                use_realsense_cfg = bool(cfg["use_realsense"])
            if "kinect_use_hw_sync" in cfg:
                use_hw_sync_cfg = bool(cfg["kinect_use_hw_sync"])
            depth = cfg.get("kinect_depth_mode")
            res = cfg.get("kinect_resolution")
            fps = cfg.get("kinect_fps")
    except Exception:
        pass
    if ip:
        ip_str = ip
    kwargs = dict(
        robot_ip="" if no_robot else ip_str,
        use_kinect=not no_kinect,
        use_realsense=use_realsense_cfg,
        strict_placement_verify=strict,
        robot_family=family,
        robot_model=model_str,
    )
    if use_hw_sync_cfg is not None:
        kwargs["kinect_use_hw_sync"] = use_hw_sync_cfg
    if depth:
        kwargs["kinect_depth_mode"] = str(depth)
    if res:
        kwargs["kinect_resolution"] = str(res)
    if fps:
        kwargs["kinect_fps"] = int(fps)
    return kwargs


def main():
    families = ["ur10e", "franka", "g1", "bimanual_franka"]
    combos = [
        dict(),
        dict(no_robot=True),
        dict(no_kinect=True),
        dict(strict_verify=True),
        dict(ip="10.0.0.5"),
        dict(no_robot=True, no_kinect=True, strict_verify=True, ip="10.0.0.5"),
    ]
    checked = 0
    for family in families:
        for combo in combos:
            cfg = SparkConfig(family=family, **combo)
            got = load_profile(cfg).to_pipeline_kwargs()
            want = golden_kwargs(
                family,
                ip=combo.get("ip"),
                no_robot=combo.get("no_robot", False),
                no_kinect=combo.get("no_kinect", False),
                strict=combo.get("strict_verify", False),
            )
            # Subset check: every historical key must still match. The config
            # layer may ADD keys (kinect_master_serial, vla_* settings) beyond
            # the original server.py logic — those are allowed, so we don't
            # re-rot this oracle each time to_pipeline_kwargs() grows a key.
            mismatched = {
                k: (want[k], got.get(k)) for k in want if got.get(k) != want[k]
            }
            assert not mismatched, (
                f"MISMATCH family={family} combo={combo}: {mismatched}"
            )
            checked += 1

    # The kwargs must construct a real PipelineConfig (no stray keys).
    from spark_real.pipeline import PipelineConfig

    for family in families:
        kwargs = load_profile(SparkConfig(family=family)).to_pipeline_kwargs()
        PipelineConfig(**kwargs)

    # Each family must carry its own gripper, cameras, and home pose so the
    # one codebase respects each system's hardware via config alone.
    franka = load_profile(SparkConfig(family="franka"))
    assert franka.gripper().get("type") == "franka_hand", franka.gripper()
    assert isinstance(franka.home_config(), list) and len(franka.home_config()) == 7

    bimanual = load_profile(SparkConfig(family="bimanual_franka"))
    assert bimanual.gripper().get("type") == "dynamixel", bimanual.gripper()
    home = bimanual.home_config()
    assert isinstance(home, dict) and home.get("left") and home.get("right"), home
    camera_types = {cam.get("type") for cam in bimanual.cameras()}
    assert "zed_mini" in camera_types, camera_types

    for family in families:
        profile = load_profile(SparkConfig(family=family))
        assert isinstance(profile.cameras(), list)
        assert isinstance(profile.workspace(), dict)

    # Fold tuning is config-driven per family: the franka yaml cloth: block
    # supplies the tuned sleeve lift the cloth_fold primitive falls back to.
    assert (
        load_profile(SparkConfig(family="franka")).cloth().get("sleeve_lift_h") == 0.120
    )
    print(
        "PER-FAMILY CLOTH OK: franka cloth: block supplies fold tuning (sleeve_lift_h=0.120)"
    )

    print(f"CONFIG EQUIVALENCE OK: {checked} family/flag combos match server.py logic")
    print("PER-FAMILY CONFIG OK: gripper, cameras, and home respected per system")


def test_config_equivalence():
    """pytest entry point — runs the full equivalence + per-family checks."""
    main()


if __name__ == "__main__":
    main()
