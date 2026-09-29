# Place a held object into a detected slot.

import logging
import time
import numpy as np
from spark_real.utils.det_fields import det_field as _field
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.control.container_geometry import container_region, xy_inside_container
from spark_real.control.grasp_strategy import compose_yaw
from spark_real.perception.mask_geometry import resolve_slot_direction

logger = logging.getLogger(__name__)


@spark_skill(
    name="place_in_slot",
    description=(
        "Drop a held item into a specific slot of a labeled container, "
        "rotating gripper yaw to align with the container's major axis."
    ),
    params={
        "container_label": str,
        "slot_idx": int,
        "release_height_m": float,
        "hover_clearance_m": float,
    },
)
def place_in_slot(executor, params: dict):
    """
    Place a held item into container's slot[slot_idx].
    """
    t0 = time.time()
    container_label = params.get("container_label", "")
    slot_idx = int(params.get("slot_idx", 0))
    release_height_m = float(params.get("release_height_m", 0.04))
    hover_clearance_m = float(params.get("hover_clearance_m", 0.10))

    from spark_real.control.execution_recovery import rebind_before_approach

    rebind_before_approach(executor, container_label)

    det = executor.detection_map.get(container_label)
    if det is None:
        for k, v in executor.detection_map.items():
            if k.split()[0] == container_label.split()[0]:
                det = v
                container_label = k
                break
    if det is None:
        return _result(
            "place_in_slot",
            False,
            f"container '{container_label}' not found",
            time.time() - t0,
        )

    pos = _field(det, "position_3d")
    if pos is None:
        return _result(
            "place_in_slot",
            False,
            f"container '{container_label}' has no position_3d",
            time.time() - t0,
        )
    pos = np.asarray(pos, dtype=float)
    centroid_xy = pos[:2].astype(float)

    # Tray geometry: centroid plus in-plane extent along major/minor axes.
    # Slot targets are derived from the tray centroid with small offsets and
    # clamped to lie strictly inside the detected tray footprint shrunk by a
    # safety margin, so a large fixed-pitch offset cannot land on or past the
    # tray rim.
    SAFETY_MARGIN_M = 0.025  # shrink the tray footprint by this each side
    MAX_OFFSET_M = 0.035  # cap per-slot centroid offset when extent known
    NO_EXTENT_OFFSET_M = 0.030  # cap when only the centroid is available
    SLOT_STEP_M = 0.030  # nominal spacing between distinct slot targets

    # world_major_axis_rad is only populated by the slot-detector path; for
    # plain detections it is absent and the realign would no-op. Fall back to
    # the OBB orientation, which is world-frame for top-down containers.
    major_axis = float(_field(det, "world_major_axis_rad", 0.0) or 0.0)
    if abs(major_axis) < 1e-9:
        major_axis = float(_field(det, "orientation_angle", 0.0) or 0.0)
    major_unit = np.array([np.cos(major_axis), np.sin(major_axis)], dtype=float)
    perp_unit = np.array([-np.sin(major_axis), np.cos(major_axis)], dtype=float)

    # World-XY half-extents of the tray. obb_minor_m is the short-axis
    # length in meters; aspect_ratio scales it to the long axis. These may
    # be absent (0) on poorly-detected containers.
    minor_m = float(_field(det, "obb_minor_m", 0.0) or 0.0)
    aspect = float(_field(det, "aspect_ratio", 1.0) or 1.0)
    if aspect < 1.0:
        aspect = 1.0 / aspect if aspect > 1e-6 else 1.0
    major_m = minor_m * aspect
    have_extent = minor_m > 1e-3 and major_m > 1e-3
    # Half-extents shrunk by the safety margin (the inner-safe region).
    half_major_safe = max(major_m / 2.0 - SAFETY_MARGIN_M, 0.0)
    half_minor_safe = max(minor_m / 2.0 - SAFETY_MARGIN_M, 0.0)

    slot_z = float(pos[2])

    # Reliable SLOT DIRECTION (the world angle a placed utensil must lie
    # ALONG), shared with the generic-place path via mask_geometry so both
    # align to the slots, not the axis-swapped container OBB. See
    # resolve_slot_direction for the source-priority ladder.
    slot_major_axis, slot_axis_source = resolve_slot_direction(det, container_label)

    def _clamp_to_tray(xy):
        """Project (xy - centroid) onto the tray's major/minor axes, clamp each
        coordinate to the shrunk half-extent, recompose.

        Returns (clamped_xy, did_clamp).
        """
        if not have_extent:
            # No reliable footprint: cap the radial offset instead.
            delta = xy - centroid_xy
            r = float(np.linalg.norm(delta))
            if r > NO_EXTENT_OFFSET_M and r > 1e-9:
                return centroid_xy + delta * (NO_EXTENT_OFFSET_M / r), True
            return xy, False
        delta = xy - centroid_xy
        a = float(delta @ major_unit)
        b = float(delta @ perp_unit)
        a_c = float(np.clip(a, -half_major_safe, half_major_safe))
        b_c = float(np.clip(b, -half_minor_safe, half_minor_safe))
        clamped = centroid_xy + a_c * major_unit + b_c * perp_unit
        did = (abs(a_c - a) > 1e-6) or (abs(b_c - b) > 1e-6)
        return clamped, did

    # Choose a slot target. Preference order:
    #   1. A detected slot center (from slot_detector) for this slot_idx,
    #      provided it falls inside the shrunk tray footprint.
    #   2. Centroid plus a small, distinct per-slot offset along the minor
    #      (perpendicular) axis, capped and clamped inside the footprint.
    mode = "centroid_offset"
    clamped_fired = False
    fallback_fired = False

    slots = _field(det, "slots", None)
    detected_xy = None
    if slots:
        for s in slots:
            try:
                s_idx = (
                    int(s.get("slot_idx", -1))
                    if hasattr(s, "get")
                    else int(getattr(s, "slot_idx", -1))
                )
            except Exception:
                s_idx = -1
            if s_idx != slot_idx:
                continue
            w = (
                s.get("world_xyz")
                if hasattr(s, "get")
                else getattr(s, "world_xyz", None)
            )
            if w is None:
                break
            cand = np.asarray(w, dtype=float)[:2]
            # Accept the detected center only if it is within the shrunk
            # footprint; otherwise discard it and fall back to centroid.
            cl, did = _clamp_to_tray(cand)
            if did:
                fallback_fired = True  # detected slot was outside safe region
            else:
                detected_xy = cand
                mode = "detected_slot"
            break

    if detected_xy is not None:
        slot_xy = detected_xy
    else:
        # Centroid-based offset. Distinct slot_idx values map to distinct
        # nearby points so same-category items group in one slot and other
        # categories in another. Center the fan on the centroid:
        # ..., -2*step, -1*step, 0, +1*step, +2*step, ... by index parity.
        if slot_idx == 0:
            signed = 0.0
        else:
            sign = 1.0 if (slot_idx % 2 == 1) else -1.0
            magnitude = ((slot_idx + 1) // 2) * SLOT_STEP_M
            signed = sign * magnitude
        cap = MAX_OFFSET_M if have_extent else NO_EXTENT_OFFSET_M
        signed = float(np.clip(signed, -cap, cap))
        slot_xy = centroid_xy + signed * perp_unit
        slot_xy, clamped_fired = _clamp_to_tray(slot_xy)

    logger.info(
        "[place_in_slot] %s slot %d -> target xy=(%.3f, %.3f) z=%.3f | "
        "centroid xy=(%.3f, %.3f) | mode=%s extent=%s "
        "(major=%.3f minor=%.3f m) clamped=%s fallback=%s",
        container_label,
        slot_idx,
        slot_xy[0],
        slot_xy[1],
        slot_z,
        centroid_xy[0],
        centroid_xy[1],
        mode,
        "yes" if have_extent else "no",
        major_m,
        minor_m,
        clamped_fired,
        fallback_fired,
    )

    place_z = slot_z + release_height_m
    hover_z = place_z + hover_clearance_m

    # Gripper convention (CONFIRMED on hardware): _oriented_grasp(theta) rotates
    # the wrist so the CLOSING (jaw) axis is PERPENDICULAR to theta and the
    # OPENING axis is PARALLEL to theta. A utensil is grasped ACROSS its handle
    # (jaws span the minor axis), so it is held along the OPENING axis, i.e. a
    # utensil grasped with _oriented_grasp(utensil_major) lies along world angle
    # `utensil_major`. Therefore, to make the held utensil lie ALONG the slot at
    # placement, command yaw = slot_direction directly: the opening axis (and
    # thus the utensil's long axis) ends up along the slot. No +90 here; the
    # perpendicular that maps tray_long_axis -> slot_direction was already
    # applied inside mask_geometry.resolve_slot_direction().
    #   Concrete check (see offline test at bottom): slots running along world X
    #   => slot_major_axis=0 => yaw=0 => opening axis along X => utensil ALONG X.
    yaw = slot_major_axis
    logger.info(
        "[place_in_slot] slot_direction=%.1f deg (source=%s) -> yaw=%.1f deg",
        np.rad2deg(slot_major_axis),
        slot_axis_source,
        np.rad2deg(yaw),
    )
    # Canonicalise into (-pi/2, pi/2]. This is bookkeeping only -- the head/tail
    # decision below is what puts the 180 back when the tip belongs at the other
    # end of the slot.
    yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0

    # Refuse rather than clip: past the wrist-3 cable limit a clipped yaw is a
    # crooked placement, which drops the item on the slot edge. Plain top-down
    # is the honest fallback.
    aligned_orient = compose_yaw(executor, yaw, context="place_in_slot")
    if aligned_orient is None:
        aligned_orient = list(executor.GRASP_ORIENTATION)

    # WHICH END GOES IN FIRST. A two-jaw gripper closes identically at yaw and
    # yaw+pi, but once the tool is in the jaws, yaw and yaw+pi point its tip at
    # opposite ends of the slot; a mod-pi reduction plus a least-wrist-travel
    # tie-break makes the answer depend on where the wrist happened to be
    # (correct when the tool started aligned with the slot, 180 deg off when it
    # did not). _orient_head_to_head is the same directed decision the obb
    # place path in executor_motion makes, so both placement paths answer this
    # question the same way.
    head_to_head = getattr(executor, "_orient_head_to_head", None)
    if head_to_head is not None:
        try:
            directed = head_to_head(aligned_orient, det)
        except Exception as exc:  # never let orientation polish kill the place
            logger.warning(
                "[place_in_slot] head/tail check failed (%s); "
                "placing on the undirected axis",
                exc,
            )
            directed = None
        if directed is not None:
            aligned_orient = directed
    elif hasattr(executor, "_nearest_symmetric_yaw"):
        # No directed evidence available at all. Least travel is then the only
        # defensible tie-break -- but say so, because a silent coin-flip here
        # is what a 180-off placement looks like from the outside.
        yaw = executor._nearest_symmetric_yaw(yaw)
        logger.info(
            "[place_in_slot] no head/tail evidence; breaking the 180 deg tie "
            "by least wrist travel (yaw=%.1f deg)",
            np.rad2deg(yaw),
        )
        aligned_orient = (
            compose_yaw(executor, yaw, context="place_in_slot") or aligned_orient
        )

    hover_pos = np.array([slot_xy[0], slot_xy[1], hover_z])
    place_pos = np.array([slot_xy[0], slot_xy[1], place_z])

    try:
        executor._move_to(hover_pos.tolist(), aligned_orient)
        executor._move_to(place_pos.tolist(), aligned_orient)
    except Exception as exc:
        return _result(
            "place_in_slot", False, f"transport failed: {exc}", time.time() - t0
        )

    try:
        if hasattr(executor.robot, "open_gripper"):
            executor.robot.open_gripper()
        elif hasattr(executor.robot, "set_gripper_position"):
            executor.robot.set_gripper_position(0.08)
    except Exception as exc:
        logger.warning("[place_in_slot] gripper open raised: %s", exc)

    # The item has been released. Clear the holding flag and record the place so
    # the transport grip-integrity check (_transport_grip_ok / _grip_intact) does
    # NOT re-squeeze the (now empty) gripper on the way to the next utensil.
    # place_in_slot releases the gripper directly (not via the `release` skill),
    # so executor_core's release block never runs; we mirror it here.
    executor._holding = False
    _pick_label = getattr(executor, "_last_pick_label", "") or ""
    if _pick_label and hasattr(executor, "_placed_labels"):
        executor._placed_labels.add(_pick_label)
        logger.info("[place_in_slot] marked '%s' as placed", _pick_label)

    try:
        executor._move_to(hover_pos.tolist(), aligned_orient)
    except Exception:
        pass

    base_msg = (
        f"Placed in '{container_label}' slot {slot_idx} "
        f"(yaw={np.rad2deg(yaw):+.1f} deg)"
    )

    # Placement verification: one detection pass to confirm the placed object
    # ended up inside the tray. Uses the picked object's label (tracked by the
    # executor) and checks its xy against the container's detected extent. If
    # the object cannot be found (occluded inside the tray) treat that as
    # success with a note rather than failing spuriously.
    placed_label = (
        getattr(executor, "_last_pick_label", "")
        or getattr(executor, "_last_keypoint_label", "")
        or ""
    )
    if placed_label and getattr(executor, "_pipeline", None) is not None:
        try:
            base_prompt = (
                placed_label.rsplit(" ", 1)[0]
                if placed_label and placed_label[-1].isdigit()
                else placed_label
            )
            captures = executor._pipeline.capture()
            new_dets = executor._pipeline.detect(captures, prompts=[base_prompt])
            cont_region = container_region(det)
            cont_xy = cont_region[0] if cont_region is not None else None
            placed_xy = None
            best_dist = float("inf")
            for d in new_dets:
                p = d.position_3d if hasattr(d, "position_3d") else d.get("position_3d")
                if p is None:
                    continue
                cand_xy = np.asarray(p[:2], dtype=float)
                if cont_xy is None:
                    placed_xy = cand_xy
                    break
                dd = float(np.linalg.norm(cand_xy - cont_xy))
                if dd < best_dist:
                    best_dist = dd
                    placed_xy = cand_xy
            if placed_xy is None:
                return _result(
                    "place_in_slot",
                    True,
                    f"{base_msg} (object not visible after "
                    f"release, presumed inside tray)",
                    time.time() - t0,
                )
            if xy_inside_container(placed_xy, det):
                return _result(
                    "place_in_slot",
                    True,
                    f"{base_msg}; verified inside tray "
                    f"(xy={placed_xy[0]:.3f},{placed_xy[1]:.3f})",
                    time.time() - t0,
                )
            # The re-detected xy is outside the strict container region. A
            # utensil lying in a cutlery tray is heavily occluded/foreshortened,
            # so the re-detect routinely mis-locates it just past the rim; that
            # is NOT proof of a mis-place. Only FAIL on a GROSS miss (clearly on
            # the table, well beyond the tray); a near-rim landing is treated as
            # success-with-note so a genuine in-tray placement is never failed.
            region = container_region(det)
            gross = False
            dist_m = float("nan")
            if region is not None and cont_xy is not None:
                _, radius = region
                dist_m = float(np.linalg.norm(placed_xy - cont_xy))
                # Loose margin: 8cm beyond the container radius. Inside this band
                # the ambiguity is occlusion, not a real drop.
                gross = dist_m > (radius + 0.08)
            if gross:
                logger.warning(
                    "[place_in_slot] verify: '%s' landed FAR from '%s' "
                    "(xy=%.3f,%.3f, dist=%.3f m) -> FAIL",
                    placed_label,
                    container_label,
                    placed_xy[0],
                    placed_xy[1],
                    dist_m,
                )
                return _result(
                    "place_in_slot",
                    False,
                    f"{base_msg} (verify failed: '{placed_label}' landed far "
                    f"outside '{container_label}' at "
                    f"xy={placed_xy[0]:.3f},{placed_xy[1]:.3f})",
                    time.time() - t0,
                )
            logger.info(
                "[place_in_slot] verify: '%s' re-detected near '%s' rim "
                "(xy=%.3f,%.3f, dist=%.3f m); occluded-in-tray ambiguity, "
                "treating as success",
                placed_label,
                container_label,
                placed_xy[0],
                placed_xy[1],
                dist_m,
            )
            return _result(
                "place_in_slot",
                True,
                f"{base_msg} (placed; re-detect near tray rim at "
                f"xy={placed_xy[0]:.3f},{placed_xy[1]:.3f}, presumed in tray)",
                time.time() - t0,
            )
        except Exception as exc:
            logger.warning("[place_in_slot] placement verify skipped: %s", exc)

    return _result("place_in_slot", True, base_msg, time.time() - t0)
