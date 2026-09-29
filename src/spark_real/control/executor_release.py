"""
ReleaseMixin: release/placement logic for ScoreExecutor.
"""

import logging
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control import success_verifier
from spark_real.control.executor_types import ExecutionResult

logger = logging.getLogger(__name__)


class ReleaseMixin:
    """
    Release action with optional tilt-and-insert.
    """

    # ------------------------------------------------------------------
    # Wrist pitch budget, by side. Positive pitch swings the wrist-mounted
    # RealSense toward the dip and 40 deg is where its clearance runs out:
    # a physical limit on this rig, not a tuning knob; do NOT raise it.
    # Negative pitch swings the camera away and has 90 deg of room.
    #
    # Human demonstrations ("put the pen in the bin", UR10e @15Hz) release
    # at 52-64 deg from vertical, only reachable on the negative side.
    # _dip_side_orient chooses the place-approach yaw at transport time so
    # the dip lands on the negative-pitch side. No at-release 180 wrist
    # flip: it produces a near-360 wrist spin and blows the release budget.
    # ------------------------------------------------------------------
    MAX_POS_PITCH_RAD = float(np.deg2rad(40))
    MAX_NEG_PITCH_RAD = float(np.deg2rad(90))

    # How far past the pitch-sign boundary the dip-side yaw aims tool Y
    # (rad). The release re-derives pitch_sign from the live TCP, so a yaw
    # that lands tool-Y exactly on the boundary would let servo residuals
    # flip the sign back to the 40 deg side. 10 deg puts |tool_y_world[0]|
    # ~0.17 past zero, far above the ~0.001 rad servo residual.
    DIP_SIDE_MARGIN_RAD = float(np.deg2rad(10))

    # Raise commanded by the tilt movel, above the resolved release z (m).
    # The resolved z parks the TCP rim+2..5cm (+held drop) over the
    # container (release_height.resolve_release_z); pitching 50+ deg there
    # sweeps the held tool's tip through the rim annulus. Teleop ramps the
    # tilt at z=0.11-0.21m and flies the last 3-7cm of descent with the tilt
    # already at max. 0.10 puts the ramp at ~rim+12-21cm on the measured pen
    # geometry; the tilted insert then descends back to the resolved z.
    TILT_RAMP_RAISE_M = 0.10

    # Hold still at max tilt before the jaws open (s). Teleop reaches max
    # tilt 0.5-1.5s before the release and the gripper opens over ~0.7s with
    # the TCP motionless. Opening the instant the insert movel returns hands
    # the pen the arm's residual motion instead of a clean drop. 1.0 = middle
    # of the human window.
    TILT_SETTLE_S = 1.0

    def _dip_side_orient(self, orient, yaw_aligned: bool = False):
        """Fold the deep-dip side choice into the place-approach orientation.

        Called by MotionMixin._transport_to with the approach orientation it
        is about to command. When the upcoming release (executor_core's
        _pending_release_tilt look-ahead) asks for a dip past the positive
        budget (MAX_POS_PITCH_RAD, the wrist-camera clearance) and `orient`
        would put the dip on that side (tool_y_world[0] > 0 at the release's
        own pitch_sign test), rotate the approach yaw so the dip lands on the
        negative side and gets the 90 deg budget instead. The yaw rides the
        transport's horizontal transit, so it costs no extra wrist travel at
        the release.

        Two cases:
          * `yaw_aligned` False (bin/bowl: no slot, no shape fit, no OBB
            adopted): the yaw is free, so pick the smallest yaw that puts
            tool Y DIP_SIDE_MARGIN_RAD past the boundary, cable limit
            enforced by compose_yaw.
          * `yaw_aligned` True: the yaw carries a slot/fit/OBB alignment and
            only its 180-twin preserves it (a two-jaw grasp is symmetric mod
            180). Try _flip_yaw_180; when the twin exceeds the wrist-3 cable
            limit, keep the alignment and let the release clamp to 40 deg. A
            shallow dip beats a crooked seat.

        Returns `orient` (possibly replaced). Never raises: any read failure
        keeps the alignment untouched and the release clamps.
        """
        pend = getattr(self, "_pending_release_tilt", None)
        if not pend:
            return orient
        try:
            tilt = abs(float(pend.get("tilt_angle", 0.0) or 0.0))
        except (TypeError, ValueError, AttributeError):
            return orient
        if tilt <= self.MAX_POS_PITCH_RAD + 1e-9:
            return orient
        try:
            tool_y = Rotation.from_rotvec(
                np.asarray(orient, dtype=float)
            ).apply([0.0, 1.0, 0.0])
        except Exception as exc:  # noqa: BLE001 - never stop a transport
            logger.warning("[dip-side] could not read orient (%s); unchanged", exc)
            return orient
        if tool_y[0] <= 0.0:
            logger.info(
                "[dip-side] release wants %.0f deg; approach already on the "
                "negative-pitch side (tool_y_x=%.2f), budget %.0f deg",
                np.rad2deg(tilt), tool_y[0], np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return orient

        if yaw_aligned:
            flipped = self._flip_yaw_180(orient)
            if flipped is not None:
                logger.info(
                    "[dip-side] release wants %.0f deg > %.0f positive budget; "
                    "flipping the ALIGNED place yaw 180 (grasp-symmetric) so "
                    "the dip gets the %.0f deg negative budget",
                    np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                    np.rad2deg(self.MAX_NEG_PITCH_RAD),
                )
                return flipped
            logger.warning(
                "[dip-side] release wants %.0f deg but the aligned yaw's 180 "
                "twin exceeds the wrist-3 cable limit; keeping the alignment, "
                "the release will clamp to %.0f deg",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
            )
            return orient

        # Free yaw: aim tool Y's world heading past the +-90 deg boundary by
        # the margin, with the least wrist travel from base. The heading of
        # tool Y after a world-Z yaw psi is (delta + psi), where delta is its
        # heading at the base orientation, and pitch_sign goes negative when
        # cos(delta + psi) < 0.
        from spark_real.control.grasp_strategy import base_orientation, compose_yaw

        try:
            ty_base = Rotation.from_rotvec(base_orientation(self)).apply(
                [0.0, 1.0, 0.0]
            )
            delta = float(np.arctan2(ty_base[1], ty_base[0]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dip-side] base orientation unreadable (%s)", exc)
            return orient
        edge = np.pi / 2.0 + self.DIP_SIDE_MARGIN_RAD
        candidates = sorted(
            (
                float((s * edge - delta + np.pi) % (2 * np.pi) - np.pi)
                for s in (1.0, -1.0)
            ),
            key=abs,
        )
        for psi in candidates:
            picked = compose_yaw(self, psi, context="place-dip")
            if picked is None:
                continue  # over the cable limit; try the other side
            logger.info(
                "[dip-side] release wants %.0f deg > %.0f positive budget; "
                "free yaw -> %.1f deg so the dip gets the %.0f deg negative "
                "budget (folded into the approach, no extra motion)",
                np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
                np.rad2deg(psi), np.rad2deg(self.MAX_NEG_PITCH_RAD),
            )
            return picked
        logger.warning(
            "[dip-side] release wants %.0f deg but no dip-side yaw fits the "
            "wrist-3 cable limit; the release will clamp to %.0f deg",
            np.rad2deg(tilt), np.rad2deg(self.MAX_POS_PITCH_RAD),
        )
        return orient

    # Settle (s) between commanding the open and letting the release witness
    # sample the jaw registers. This one stays a sleep:
    #
    #   * it cannot become a condition wait. The witness reads the jaws by
    #     uploading a publish program, and a new program REPLACES the running
    #     rq_move_and_wait open, so polling the open would cancel the open and
    #     leave the object still gripped.
    #   * open_gripper does not pass a speed_norm, so the jaw speed on this
    #     rig is unmeasured, and 0.5 s (on top of the driver's own 0.5 s) is
    #     already optimistic against the datasheet's slowest 20 mm/s.
    #     Shortening it on a guess risks reporting "never released" on a
    #     release that worked.
    #
    # Env SPARK_RELEASE_OPEN_SETTLE_S lets the operator tune it after
    # measuring the real open time (the witness logs jaw_pos_before/after).
    RELEASE_OPEN_SETTLE_S = 0.5

    def _release_open_settle_s(self) -> float:
        raw = os.environ.get("SPARK_RELEASE_OPEN_SETTLE_S")
        if raw is None:
            return float(self.RELEASE_OPEN_SETTLE_S)
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("SPARK_RELEASE_OPEN_SETTLE_S=%r ignored", raw)
            return float(self.RELEASE_OPEN_SETTLE_S)

    def _transport_delivered(self) -> tuple:
        """Did the transport this release depends on put the TCP over the container?

        Returns ``(ok, detail)``. ``ok`` is True whenever there is nothing to
        judge: not holding anything, no place transport ran, the record
        belongs to an earlier task (``_last_place_label`` is reset per task and
        must still match), or the container's extent is unknown. The gate only
        ever fires on positive evidence that the TCP is outside the container
        it was sent to; see MotionMixin._place_arrival for the tolerance.

        The verdict is recomputed here, against the TCP as it stands with the
        jaws about to open; the stored record only says which container a
        place transport was aimed at. A recovery may have re-approached and
        fixed it, or a move may have drifted off the container after a clean
        arrival.
        """
        if not self._holding:
            return True, ""
        record = getattr(self, "_place_arrival_record", None)
        if not record:
            return True, ""
        label = record.get("label") or ""
        if not label or label != getattr(self, "_last_place_label", ""):
            return True, ""
        return self._place_arrival(label, self.detection_map.get(label))


    # ------------------------------------------------------------------
    # Proprioceptive release verdict
    #
    # The arm's own evidence about whether the place worked, collected in the
    # seconds around the jaw opening. Camera verification is weakest where
    # these signals are strongest: a filled slot no longer looks like a slot,
    # and a tray of identical sockets defeats label re-matching. Every signal
    # reports pass/fail/None (None = could not measure, never a guess); the
    # aggregate is "fail" on any hard contradiction, "pass" when everything
    # measurable agrees, else "uncertain".
    # ------------------------------------------------------------------

    def _collect_release_proprio(self, commanded_orient=None):
        v = {
            "commanded_pitch_achieved": None,
            "carry_force_n": None,
            "jaws_opened": None,       # part 2 fills these two
            "weight_dropped": None,
        }
        # Commanded vs achieved orientation. This catches a dropped motion
        # command (a tilt movel that was silently a no-op while the log said
        # 'completed'). 0.15 rad (~9 deg) is far above servo residuals
        # (~0.001) and far below the commanded 32 deg tilt.
        if commanded_orient is not None:
            try:
                actual = self.current_orientation()
                if actual is not None:
                    r_cmd = Rotation.from_rotvec(list(commanded_orient))
                    r_act = Rotation.from_rotvec(list(actual))
                    err = float((r_cmd.inv() * r_act).magnitude())
                    v["commanded_pitch_achieved"] = bool(err < 0.15)
                    v["orient_err_rad"] = round(err, 4)
            except Exception as exc:  # noqa: BLE001 - verdict must not block release
                logger.warning("release proprio: orientation read failed: %s", exc)
        # Carried load just before the jaws open, for the weight-drop
        # comparison in part 2. Averaged briefly because the UR force
        # estimate is ~+-2 N noisy; a light pen stays below the floor and the
        # signal reports None rather than guessing.
        try:
            fz = []
            for _ in range(5):
                fz.append(float(self.robot.get_tcp_force()[2]))
                time.sleep(0.02)
            v["carry_force_n"] = round(float(np.mean(fz)), 2)
        except Exception:
            pass
        return v

    _PROPRIO_WEIGHT_FLOOR_N = 1.5  # below this the drop is inside sensor noise

    def _finish_release_proprio(self):
        v = getattr(self, "_release_proprio", None)
        if v is None:
            return
        # Jaw evidence comes from the witness the verifier just finished; keyed
        # on the same record verify_scope.still_holding reads.
        w = getattr(self, "_release_witness", None)
        released = None
        if w is not None:
            released = getattr(w, "released", None)
            if released is None and isinstance(w, dict):
                released = w.get("released")
        v["jaws_opened"] = released
        # Weight after the open: carried minus current should be about the
        # object's weight. Only meaningful when the carried reading existed
        # and the difference clears the noise floor in SOME direction.
        try:
            if v.get("carry_force_n") is not None:
                fz = []
                for _ in range(5):
                    fz.append(float(self.robot.get_tcp_force()[2]))
                    time.sleep(0.02)
                drop = float(v["carry_force_n"] - float(np.mean(fz)))
                v["weight_drop_n"] = round(drop, 2)
                if abs(drop) >= self._PROPRIO_WEIGHT_FLOOR_N:
                    v["weight_dropped"] = bool(drop < 0)
        except Exception:
            pass
        # Aggregate. Hard contradictions fail; all-green passes; anything
        # unmeasurable stays uncertain rather than optimistic.
        hard_fail = (
            v.get("jaws_opened") is False
            or v.get("commanded_pitch_achieved") is False
        )
        measured = [
            x for x in (v.get("jaws_opened"), v.get("commanded_pitch_achieved"),
                        v.get("weight_dropped"))
            if x is not None
        ]
        v["verdict"] = (
            "fail" if hard_fail
            else ("pass" if measured and all(measured) else "uncertain")
        )
        logger.info("[proprio-verdict] %s | %s", v["verdict"],
                    {k: x for k, x in v.items() if k != "verdict"})

    def _release(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()

        # Arrival gate. Opening the jaws is irreversible, so it is gated on the
        # object actually being over the container, and the failure is
        # reported (failed action + the transport verify gate) rather than
        # swallowed: an arm that silently refuses to let go and holds the
        # object forever is worse than the drop. executor_core's failure path
        # runs recovery, then force-verifies the grip and keeps the object
        # rather than dumping it wherever the arm happens to be.
        delivered, why = self._transport_delivered()
        if not delivered:
            success_verifier.set_gate(self, "transport", False, why)
            logger.warning("Release REFUSED, transport did not arrive: %s", why)
            return ExecutionResult(
                action_type="release",
                success=False,
                message=f"Refused to open the jaws: {why}",
                duration=time.time() - t0,
            )

        release_tcp = self._get_current_position()
        logger.info(
            "Release TCP: [%.3f, %.3f, %.3f] (z=%.3f)",
            release_tcp[0],
            release_tcp[1],
            release_tcp[2],
            release_tcp[2],
        )

        tilt_angle = abs(params.get("tilt_angle", 0.0))
        tilted = False
        signed_angle = 0.0

        if tilt_angle > 0.01:
            gemini_pitch = float(params.get("pitch_sign", 1.0))
            pitch_sign = gemini_pitch

            try:
                obs = self.robot.get_observation()
                tcp_pose = obs["tcp_pose"]
                current_rot = Rotation.from_rotvec(tcp_pose[3:6])
                tool_y_world = current_rot.apply([0, 1, 0])
                pitch_sign = 1.0 if tool_y_world[0] > 0 else -1.0
                logger.info(
                    "Pitch sign from TCP: tool_Y_world=(%.2f,%.2f,%.2f) "
                    "obj extends in X=%.2f -> pitch_sign=%.0f (gemini hint=%.0f)",
                    *tool_y_world,
                    tool_y_world[0],
                    pitch_sign,
                    gemini_pitch,
                )
            except Exception as e:
                logger.warning(
                    "Could not read TCP for pitch sign: %s, using Gemini hint", e
                )
                pitch_sign = gemini_pitch

            MAX_POS_PITCH = self.MAX_POS_PITCH_RAD
            MAX_NEG_PITCH = self.MAX_NEG_PITCH_RAD

            # Tilt onto the orientation the arm is actually holding, not onto
            # the bare top-down constant: the current TCP rotation carries the
            # yaw the place just aligned, and pitching it about the tool X
            # keeps the alignment and adds the dip.
            try:
                _obs = self.robot.get_observation()
                base_rot = Rotation.from_rotvec(_obs["tcp_pose"][3:6])
                logger.info(
                    "Tilt base: current TCP orientation (keeps the placement yaw)"
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning(
                    "Could not read TCP for the tilt base (%s); falling back to "
                    "GRASP_ORIENTATION, which DROPS any placement yaw", _e
                )
                base_rot = Rotation.from_rotvec(self.GRASP_ORIENTATION)

            # Positive pitch past its limit: clamp, never flip. A 180 wrist
            # flip about z to pitch negative instead costs a full extra wrist
            # revolution and can blow the release budget mid-insert. Excess is
            # clamped by the shared min() below and logged here so a plan
            # asking for the moon is visible without being obeyed.
            #
            # Deep dips are not lost by this clamp: _dip_side_orient chose the
            # place-approach yaw at transport time so a >40 deg request lands
            # on the negative side (90 deg budget) with pitch_sign=-1 and this
            # branch never fires. Reaching it means that choice was impossible
            # (an aligned yaw whose 180 twin exceeds the cable limit, or no
            # transport look-ahead ran); then a shallow 40 deg dip is the
            # right answer, not an at-release reorientation.
            if pitch_sign > 0 and tilt_angle > MAX_POS_PITCH:
                logger.info(
                    "Positive pitch %.0f deg exceeds limit (%.0f); clamping "
                    "(no 180 flip; the dip-side yaw choice was not available)",
                    np.rad2deg(tilt_angle),
                    np.rad2deg(MAX_POS_PITCH),
                )

            if pitch_sign > 0:
                tilt_angle = min(tilt_angle, MAX_POS_PITCH)
            else:
                tilt_angle = min(tilt_angle, MAX_NEG_PITCH)

            signed_angle = -pitch_sign * tilt_angle
            tilt = Rotation.from_euler("x", signed_angle)
            tilted_rot = base_rot * tilt
            tilted_orient = tilted_rot.as_rotvec().tolist()
            logger.info("Tilted release: pitch %+.0f deg", np.rad2deg(signed_angle))

            # Keep the dipping tip over the spot the transport delivered.
            # The tip retracts toward the TCP by half_length*(1-cos) as the
            # tool tilts, so the TCP shifts the same amount along the tool-Y
            # horizontal heading (the direction the object's far end points);
            # a world-X shift would walk the tip sideways off the container
            # under the dip-side yaw. Reduces to pitch_sign*X at zero yaw.
            # Teleop dxy during the dip is small (<=8cm across episodes);
            # this gives 1.9cm at 40 deg.
            half_length = 0.08
            ty_xy = base_rot.apply([0.0, 1.0, 0.0])[:2]
            ty_norm = float(np.linalg.norm(ty_xy))
            tip_dir = (
                ty_xy / ty_norm if ty_norm > 1e-6 else np.array([pitch_sign, 0.0])
            )
            offset_xy = tip_dir * half_length * (1 - np.cos(tilt_angle))
            # Tilt high, not at the release z. The tilt movel raises the TCP
            # by TILT_RAMP_RAISE_M while the tilt ramps (teleop shows the tilt
            # at z=0.11-0.21m, never at the rim); the ~10cm of travel at 0.3x
            # velocity stretches the ramp to the human's 2.5-3s. The tilted
            # insert below then descends back to the resolved release z.
            tilt_pos = release_tcp.copy()
            tilt_pos[0] += offset_xy[0]
            tilt_pos[1] += offset_xy[1]
            tilt_pos[2] += self.TILT_RAMP_RAISE_M
            logger.info(
                "Tilt TCP offset: half_len=%.0fmm, offset_xy=(%+.0f,%+.0f)mm, "
                "raise=%.0fmm -> tilt_pos=(%.3f,%.3f,%.3f)",
                half_length * 1000,
                offset_xy[0] * 1000,
                offset_xy[1] * 1000,
                self.TILT_RAMP_RAISE_M * 1000,
                *tilt_pos,
            )

            try:
                logger.info("Executing tilt movel (ramp at height)...")
                self._move_to_linear(
                    tilt_pos, tilted_orient, velocity=self.velocity * 0.3
                )
                logger.info("Tilt movel completed")
                tilted = True
            except Exception as e:
                logger.warning("Tilt failed: %s - releasing straight", e)
                tilted = False

            if tilted:
                # Descend, tilted, back to the resolved release z: the z the
                # place transport arrived at, which release_height already
                # bounded to keep the TCP off the rim plane (teleop: 12-20cm
                # of z during the dip, jaws opening at the rim plane). Known
                # conservatism: a tilted tool hangs cos(tilt) of its overhang,
                # so the object's bottom ends a couple of cm higher than the
                # vertical-drop geometry aimed; the rim guard is the harder
                # constraint.
                insert_pos = self._get_current_position()
                insert_pos[2] = max(release_tcp[2], self.TABLE_Z_FLOOR + 0.02)
                logger.info(
                    "Inserting tilted: descending to resolved z=%.3f "
                    "(%.0fmm below the ramp)",
                    insert_pos[2],
                    (tilt_pos[2] - insert_pos[2]) * 1000,
                )
                try:
                    self._move_to_linear(
                        insert_pos, tilted_orient, velocity=self.velocity * 0.2
                    )
                    logger.info("Insert descent completed")
                except Exception as e:
                    logger.warning("Insert descent failed: %s", e)
                # Hold still at max tilt before the jaws open (see
                # TILT_SETTLE_S). Abort-aware like every other release sleep.
                self._abort_sleep(self.TILT_SETTLE_S)

        # G3 release witness: read the grip immediately before opening, and
        # again after the settle. held_before == False means the object was
        # already gone; the run fails with no capture and no SAM3, which is
        # the cheap catch for a drop that happened mid-transport.
        _witness_state = success_verifier.begin_release_witness(self)
        # Proprioceptive verdict, part 1: sample what the arm itself can attest
        # before the jaws open (commanded pose reached, commanded tilt
        # achieved, force carried).
        self._release_proprio = self._collect_release_proprio(
            commanded_orient=(tilted_orient if tilted else None),
        )
        self.robot.open_gripper()
        # This settle is load-bearing, not padding: the witness reads the jaw
        # registers by uploading a publish program, which would REPLACE the
        # rq_move_and_wait open still running on the controller. open_gripper
        # sleeps 0.5 s of its own, so the jaws are parked by the time the
        # witness samples them. It then confirms by polling, not by waiting
        # longer (success_verifier._confirm_jaws_open). See
        # RELEASE_OPEN_SETTLE_S for why this one keeps its full duration.
        self._abort_sleep(self._release_open_settle_s())
        # Clear _holding FIRST: on a rig with no jaw register, the witness
        # falls back to this flag, and reading it while still True would
        # report the object as never released on every successful place.
        self._holding = False
        # Consumed: the arrival question this record answered is now settled.
        self._place_arrival_record = None
        success_verifier.finish_release_witness(
            self, _witness_state, self._last_place_label
        )
        # Proprioceptive verdict, part 2: the witness has now read the jaws.
        self._finish_release_proprio()

        # Retract straight up in the current orientation. A release is a grip
        # action and must not command a yaw: reorienting the wrist after the
        # jaws are open is wasted motion, and snapping to GRASP_ORIENTATION
        # from a tilted place pose can swing the wrist ~70-85 deg and overrun
        # the release timeout budget. The next task's pre-home restores the
        # base orientation anyway. Read the current orientation via
        # get_observation() (the normalized cross-family pose interface); if
        # it cannot be read, do not guess a different one.
        orient = None
        try:
            obs = self.robot.get_observation()
            # `a or b` on a numpy array raises "truth value of an array with
            # more than one element is ambiguous"; compare to None.
            tcp = obs.get("tcp_pose")
            if tcp is None:
                tcp = obs.get("tcp_pos")
            if tcp is not None and len(tcp) >= 6:
                orient = list(np.array(tcp[3:6], dtype=float))
                logger.info(
                    "Release retract: holding the CURRENT orientation %s",
                    [round(float(v), 3) for v in orient],
                )
            else:
                logger.warning(
                    "Release retract: get_observation gave no tcp_pose (%r); "
                    "falling back to GRASP_ORIENTATION, which will SNAP the "
                    "wrist back to base after the release",
                    list(obs.keys()) if isinstance(obs, dict) else type(obs),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Release retract: could not read the current orientation (%s); "
                "falling back to GRASP_ORIENTATION -- expect a wrist snap", exc
            )
        if orient is None:
            logger.warning(
                "Release retract: current orientation unreadable; skipping the "
                "reorientation entirely rather than snapping to base"
            )
        current = self._get_current_position()
        retract = current.copy()
        # Retract height is a parameter so the opening jaws clear neighbouring
        # objects (a couple of centimetres) without stacking on a lift the
        # plan adds itself: 8 cm plus a move_relative dz 0.05 is 13 cm up
        # against compliant_push's 6 cm of travel.
        retract_z = float(params.get("retract_z", 0.08))
        retract[2] += retract_z
        if abs(retract_z - 0.08) > 1e-6:
            logger.info("Release retract: %.3f m (planner-set)", retract_z)
        if orient is not None and abs(retract_z) > 1e-6:
            self._move_to(retract, orient)
        elif orient is None and abs(retract_z) > 1e-6:
            # The orientation read failed, but the lift must still happen:
            # skipping it leaves the open jaws parked centimeters beside the
            # just-placed object, a worse failure than the wrist snap. Try the
            # executor's own TCP read (a different path than get_observation)
            # before paying the snap to base.
            fb = None
            cur = getattr(self, "current_orientation", None)
            if callable(cur):
                fb = cur()
            self._move_to(
                retract, list(fb) if fb is not None else list(self.GRASP_ORIENTATION)
            )
        msg = f"Released at z={self._get_current_position()[2]:.3f}"
        if tilted:
            msg += f" (tilted {np.rad2deg(signed_angle):+.0f} deg, inserted)"

        # A release that did not open the jaws is a failed release. Returning
        # success here lets the tree go on to close_gripper + compliant_push
        # and press the still-held tool into the bench (~89 N) until the
        # controller protective-stops. Reporting the failure routes the same
        # situation into recovery instead, which lifts.
        #
        # Positive evidence only: `released is False` means the jaws were
        # measured shut. An unknown (None) still passes, because the registers
        # are known to read stale and a false failure here would strand a
        # perfectly good place.
        witness = getattr(self, "_release_witness", None)
        released = None
        if witness is not None:
            released = getattr(witness, "released", None)
            if released is None and isinstance(witness, dict):
                released = witness.get("released")
        if released is False:
            logger.warning(
                "Release FAILED: the jaws never opened (%s). Reporting failure "
                "so recovery runs -- continuing here is what pressed a held "
                "tool into the bench on 2026-08-24.", msg,
            )
            return ExecutionResult(
                action_type="release",
                success=False,
                message="Release did not open the jaws; still holding",
                duration=time.time() - t0,
            )
        return ExecutionResult(
            action_type="release", success=True, message=msg, duration=time.time() - t0
        )
