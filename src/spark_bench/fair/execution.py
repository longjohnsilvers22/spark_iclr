"""
Per-episode execution: plan dispatch, motion, retract, frame recording.
"""
from __future__ import annotations

import time
import yaml

try:
    import h5py
except ImportError:
    h5py = None

from .config import (
    FairConfig,
    Optional,
    _get_sam3,
    _library_replay_plan,
    bt_replay_enabled,
    detect_scene,
    detect_scene_privileged,
    execute_on_libero,
    get_library,
    mujoco,
    np,
    os,
    redetect_agentview,
    select_prompts,
    shadow_enabled,
    shadow_k,
    shadow_select_and_execute,
)
from .planning import (
    _gated_plan,
    _plan,
    _plan_diverse_candidates,
    _run_label_validator,
)

# Imported after .config so the MUJOCO_GL / allocator env defaults are in
# place before the executor pulls in mujoco / torch transitively.
from spark_real.control.attribution import Layer, attribute_failure
from spark_bench.libero_pro.planning import _bt_frozen
from spark_bench.libero_pro.executor import _flatten_tree
from spark_bench.libero_pro.perception import (
    annotation_rescue,
    disambiguate_pick_instance,
)
from spark_real.perception.sticky_binding import compute_scene_diff, fuzzy_key


def _execute(env, score, det, instruction: str, cfg: FairConfig, sam3,
              prompts: Optional[list] = None,
              trial_meta: Optional[dict] = None) -> None:
    """
    Common execute-via-the-executor call (used by primary + recovery paths).

    ``prompts`` (when given) is the active SAM3 detection prompt list.  The
    executor only reads it when ``cfg.freshness_timeout_s`` is set; in that
    case it re-runs SAM3 with the same vocabulary between primitives to
    bound perception staleness (DynamicVLA LAAS analog).

    ``trial_meta`` receives the executor's per-trial event records
    (grasp_outcomes, capture_events, scene_diffs, retarget_events).
    """
    execute_on_libero(env, score, det.det_map, cfg,
                       sam3=sam3, depth=det.depth,
                       cam_pos=det.cam.pos, cam_mat=det.cam.mat,
                       cam_fovy=det.cam.fovy,
                       cam_w=cfg.cam_width, cam_h=cfg.cam_height,
                       instruction=instruction,
                       prompts_for_refresh=prompts,
                       trial_meta=trial_meta)



def _oracle_gate(env) -> bool:
    """Mid-episode read of the simulator's success predicate, gated.

    Under ``SPARK_NO_ORACLE=1`` every CONTROL read of ``env.check_success``
    returns False: the robot never learns from the simulator that it is done,
    that it should stop pushing, or that a retry is unnecessary. Scoring is
    untouched; the trial's outcome is still the official predicate, read once
    after the last action. This exists to MEASURE what the mid-episode reads
    are worth, because the paper criticises Zetta for exactly this access.
    """
    if os.environ.get('SPARK_NO_ORACLE', '').strip() in ('1', 'true', 'yes'):
        return False
    try:
        return bool(env.check_success())
    except Exception:
        return False


def _retract_arm(env, model, data, ee_site: int) -> None:
    """
    Lift the arm 10 cm before re-detecting (recovery loop helper).
    """
    mujoco.mj_forward(model, data)
    ee = (data.site_xpos[ee_site].copy() if ee_site >= 0
          else np.array([0.0, 0.0, 0.3]))
    lift = ee.copy(); lift[2] = max(ee[2] + 0.10, 0.3)
    for _ in range(100):
        try:
            curr = env.env.robots[0].controller.ee_pos
        except Exception:
            break
        err = lift - curr
        if np.linalg.norm(err) < 0.01:
            break
        a = np.zeros(7); a[:3] = np.clip(err * 8 / 0.05, -1, 1); a[6] = -1
        try:
            env.step(a)
        except Exception:
            break


def _store_in_bt_library(instruction: str, score: dict, dets: list) -> None:
    """
    Voyager-style skill accumulation - store successful plans for reuse.
    """
    if get_library is None:
        return
    try:
        bt_lib = get_library()
        bt_lib.add(
            instruction=instruction,
            score=score,
            objects=[d.label for d in dets if d.position_3d is not None],
            robot='franka',
            scene_type='libero_pro',
        )
    except Exception:
        pass


def _find_ee_site(model) -> int:
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn:
            return s
    return -1


def _score_primary_label(score) -> str:
    """
    First ``move_to_keypoint`` label in the flattened score - the pick
    target, i.e. the object whose binding stability decides whether a
    failure is a perception problem.
    """
    try:
        for act in _flatten_tree((score or {}).get('tree', {})):
            lbl = (act.get('params', {}) or {}).get('keypoint_label')
            if lbl:
                return str(lbl)
    except Exception:
        pass
    return ''


def _layered_recovery(env, model, data, cfg: FairConfig, sam3,
                        score, det, instruction: str, prompts,
                        rgb, wrist_rgb, pick_hint: str, place_hint: str,
                        picks, places, pick_counts,
                        trial_meta: Optional[dict], stash_bt,
                        max_attempts: int = 3) -> bool:
    """
    ANCHOR-style minimum-responsible-layer recovery ladder.

    Per attempt: retract, re-check the goal, re-detect, diff the fresh
    detections against the previous binding, then let
    :func:`attribute_failure` pick the repair rung:

    * PERCEPTION -> re-bind (execute the SAME score against the fresh
      detections; no replan).
    * EXECUTION  -> local retry of the score (detections were stable, the
      motion failed); counts toward ``cfg.max_local_retries``.
    * PLAN       -> tier-3 replan via the configured planner (unavailable
      under --no-gemini: the ladder stops there).

    Attribution decisions land in trial_meta['recovery_attributions'],
    each stamped with ``t`` (decision) and ``t_resume`` (re-execution
    start) for the adaptation-latency metric.
    """
    meta = trial_meta if trial_meta is not None else {}
    ee_site = _find_ee_site(model)
    target_label = _score_primary_label(score) or pick_hint
    det_cur, score_cur = det, score
    local_retries = 0

    for attempt in range(max_attempts):
        _retract_arm(env, model, data, ee_site)
        if _oracle_gate(env):
            return True

        if getattr(cfg, 'privileged', False):
            obs2 = (env._get_observations()
                    if hasattr(env, '_get_observations') else {})
            det2 = detect_scene_privileged(env, prompts, obs2, cfg)
        else:
            det2 = redetect_agentview(env, sam3, prompts, cfg)
            if det2 is not None and det2.det_map:
                # Label retention: the recovery frame is agentview-only,
                # so labels the bind frame got from the wrist camera (or
                # that are occluded right now) would vanish from det_map
                # and downstream fuzzy matching would re-aim at a lookalike
                # sharing a word. Merge freshest-wins per label, retaining
                # last-known entries for labels the fresh frame did not
                # produce (same semantics as the executor's freshness merge).
                merged = dict(det_cur.det_map)
                merged.update(det2.det_map)
                det2.det_map = merged
            # Recovery re-bind is the other moment where the wrong twin
            # instance can be latched (cfg.instance_disambig).
            if det2 is not None and det2.det_map:
                try:
                    disambiguate_pick_instance(
                        det2, target_label or pick_hint, instruction, cfg,
                        sam3=sam3, trial_meta=trial_meta)
                except Exception as e:
                    if cfg.verbose:
                        print(f"[InstanceDisambig] recovery re-bind "
                              f"failed: {e}")

        sd = None
        if det2 is None or not det2.det_map:
            verify = {'predicate_ok': False, 'detection_missing': True}
        else:
            # Diff against where execution LEFT the scene when the
            # executor published it (self-caused changes re-bound), not
            # the plan-time binding - the object the robot just placed
            # must not read as an external 'moved'.
            final_binding = meta.get('final_binding')
            if final_binding:
                bound = {k: (np.asarray(v, dtype=float)
                             if v is not None else None)
                         for k, v in final_binding.items()}
            else:
                bound = {k: getattr(d, 'position_3d', None)
                         for k, d in det_cur.det_map.items()}
            current = {k: getattr(d, 'position_3d', None)
                       for k, d in det2.det_map.items()}
            mujoco.mj_forward(model, data)
            grip = (data.site_xpos[ee_site].copy() if ee_site >= 0 else None)
            sd = compute_scene_diff(
                bound, current, target_label=target_label or None,
                gripper_pos=grip,
                moved_threshold_m=float(getattr(
                    cfg, 'scene_diff_moved_cm', 1.5)) / 100.0,
                self_radius_m=float(getattr(
                    cfg, 'scene_diff_self_radius_cm', 12.0)) / 100.0)
            key = (fuzzy_key(det2.det_map, target_label)
                   if target_label else None)
            conf = (getattr(det2.det_map[key], 'confidence', None)
                    if key is not None else None)
            last_grasp = (meta.get('grasp_outcomes') or [{}])[-1]
            verify = {'predicate_ok': False,
                        'detection_missing': key is None,
                        'detection_confidence': conf,
                        'grasp_outcome': last_grasp.get('outcome')}

        layer = attribute_failure(
            verify, sd, local_retries,
            max_local_retries=int(getattr(cfg, 'max_local_retries', 1)))
        rec = {'attempt': attempt, 'layer': layer.value,
                'target_label': target_label,
                'diff_status': getattr(sd, 'target_status', None),
                'detection_confidence': verify.get('detection_confidence'),
                't': time.time()}
        meta.setdefault('recovery_attributions', []).append(rec)
        if cfg.verbose:
            print(f"[Recovery] attempt {attempt}: layer={layer.value} "
                  f"target={target_label!r} "
                  f"diff={rec['diff_status']}", flush=True)

        if layer is Layer.PLAN:
            if cfg.no_gemini:
                rec['action'] = 'replan_unavailable'
                return False
            if det2 is not None and det2.det_map:
                det_cur = det2
            score2 = _plan(cfg, instruction, det_cur.det_map,
                             getattr(det_cur, 'rgb', rgb), wrist_rgb,
                             pick_hint, place_hint, picks=picks,
                             places=places, pick_counts=pick_counts,
                             prompts=prompts)
            if not score2:
                rec['action'] = 'replan_failed'
                continue
            score2 = _run_label_validator(
                cfg, score2, det_cur.det_map, instruction,
                getattr(det_cur, 'rgb', rgb), wrist_rgb, pick_hint,
                place_hint, trial_meta=trial_meta,
                slot_key='validator_actions_recovery',
                picks=picks, places=places, pick_counts=pick_counts)
            stash_bt(trial_meta, score2, key='bt_yaml_recovery')
            score_cur = score2
            rec['action'] = 'replan'
        elif layer is Layer.PERCEPTION:
            if det2 is None or not det2.det_map:
                rec['action'] = 'redetect_failed'
                continue
            det_cur = det2
            rec['action'] = 'rebind'
        else:  # EXECUTION
            local_retries += 1
            if det2 is not None and det2.det_map:
                det_cur = det2  # stable anyway; retry off the fresh frame
            rec['action'] = 'local_retry'

        rec['t_resume'] = time.time()
        _execute(env, score_cur, det_cur, instruction, cfg, sam3,
                  prompts=prompts, trial_meta=trial_meta)
        if _oracle_gate(env):
            return True
    return False


def run_spark_on_libero_env(env, prompts: list, instruction: str,
                              pick_hint: str, place_hint: str,
                              cfg: FairConfig,
                              trial_meta: Optional[dict] = None,
                              picks: Optional[list] = None,
                              places: Optional[list] = None,
                              pick_counts: Optional[list] = None,
                              post_plan_hook=None) -> bool:
    """
    Run the SPARK pipeline once on a LIBERO env.

    Pipeline: render -> SAM3 detect -> Gemini plan -> execute -> check_success.
    On failure, retract + re-detect + re-plan + re-execute (twice).

    If ``trial_meta`` is supplied (a dict), per-trial diagnostics are
    written into it: ``bt_yaml`` (raw Gemini YAML for the first plan
    attempt), ``bt_yaml_recovery`` (list of YAML strings from recovery
    re-plans), and ``planner`` (which planner emitted the BT).  The dict
    is mutated in place - additive only, no schema break for callers
    that pass nothing.  The executor adds its own event records when the
    adaptive-execution flags are on (grasp_outcomes, capture_events,
    scene_diffs, retarget_events, recovery_attributions).

    ``post_plan_hook(env, score, det)`` (when given) fires once after
    planning + label validation, before execution - the LIBERO-Dyn
    perturbation injection point.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    def _stash_bt(meta: Optional[dict], score, key: str = 'bt_yaml') -> None:
        if meta is None or not isinstance(score, dict):
            return
        raw = score.get('__raw_yaml')
        planner = score.get('__planner')
        if key == 'bt_yaml':
            if raw is not None:
                meta['bt_yaml'] = raw
            if planner is not None:
                meta['planner'] = planner
        else:
            meta.setdefault('bt_yaml_recovery', [])
            meta['bt_yaml_recovery'].append({
                'planner': planner, 'bt_yaml': raw})

    # Early-out: goal already satisfied (e.g. "Turnoff" with stove-off init).
    # Under SPARK_NO_ORACLE the robot must act and earn the episode-end check.
    if _oracle_gate(env):
        return True

    obs, _, _, _ = env.step(np.zeros(7))
    rgb_raw = obs.get('agentview_image')
    if rgb_raw is None:
        return False
    rgb = rgb_raw[::-1].copy()
    wrist_rgb_raw = obs.get('robot0_eye_in_hand_image')
    wrist_rgb_for_llm = (wrist_rgb_raw[::-1].copy()
                          if wrist_rgb_raw is not None else None)

    prompts = select_prompts(env, rgb, wrist_rgb_for_llm,
                              instruction, cfg, prompts)

    # Stash instruction on cfg so detect_scene can see it for the
    # compound-AND multi-instance heuristic (e.g. "both moka pots" ->
    # flag "moka pot" prompt as multi_instance so SAM3 returns every
    # instance, mirroring spark_real/pipeline.py's batch-task path).
    cfg._current_instruction = instruction

    if getattr(cfg, 'privileged', False):
        det = detect_scene_privileged(env, prompts, obs, cfg)
        sam3 = None  # not needed downstream when privileged
    else:
        sam3 = _get_sam3()
        det = detect_scene(env, sam3, prompts, obs, cfg)
        # Annotation-rescue rung (cfg.annotation_rescue, default off):
        # prompts that survived every prompt-selection layer with no
        # detection get one provider point -> SAM3 click seed each.
        # Runs AFTER the text path so it only ever adds bindings; events
        # land in trial_meta['annotation_rescue']. Never breaks a trial.
        if getattr(cfg, 'annotation_rescue', False):
            try:
                annotation_rescue(det, prompts, cfg, sam3=sam3,
                                    trial_meta=trial_meta)
            except Exception as e:
                if cfg.verbose:
                    print(f"[AnnRescue] failed: {e}")
    if not det.det_map:
        if cfg.verbose:
            print("No detections")
        return False

    # Optional bind-time instance disambiguation (cfg.instance_disambig):
    # if >1 instance of the pick label is visible, the Phase-3 machinery
    # picks the one satisfying the instruction's spatial relation before
    # the binding freezes.  Never allowed to break a trial.
    if not getattr(cfg, 'privileged', False):
        try:
            disambiguate_pick_instance(det, pick_hint, instruction, cfg,
                                         sam3=sam3, trial_meta=trial_meta)
        except Exception as e:
            if cfg.verbose:
                print(f"[InstanceDisambig] failed: {e}")

    # Plan-time detection snapshot for the trace-selector's grounding
    # features. Taken after bind-time disambiguation so it matches the
    # det_map the planner actually consumes.
    if trial_meta is not None:
        try:
            trial_meta['det_summary'] = {
                lbl: {
                    'conf': float(d.confidence),
                    'pos': ([float(x) for x in d.position_3d]
                            if getattr(d, 'position_3d', None) is not None
                            else None),
                }
                for lbl, d in det.det_map.items()}
        except Exception:
            pass

    # Zero-LLM library-replay path: a high-overlap bt_library match skips
    # Gemini entirely; a miss falls through to the planner.
    score = None
    if bt_replay_enabled() and not cfg.no_gemini:
        score = _library_replay_plan(
            instruction, det.det_map,
            picks=picks, places=places, pick_counts=pick_counts)

    if score is None and getattr(cfg, 'wm_safety_gate', False):
        # Current EE position for the FK trace.
        ee_site = _find_ee_site(model)
        if ee_site >= 0:
            ee_init_xyz = np.asarray(data.site_xpos[ee_site], dtype=np.float32)
        else:
            ee_init_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        score = _gated_plan(cfg, instruction, det.det_map, rgb,
                             wrist_rgb_for_llm, pick_hint, place_hint,
                             ee_init_xyz=ee_init_xyz,
                             picks=picks, places=places,
                             pick_counts=pick_counts)
    elif score is None:
        score = _plan(cfg, instruction, det.det_map, rgb,
                       wrist_rgb_for_llm, pick_hint, place_hint,
                       picks=picks, places=places,
                       pick_counts=pick_counts, prompts=prompts)
    if score is None:
        return False
    # Hard post-parse label validator: snap or one-retry any label that
    # isn't in det_map.keys() BEFORE the executor sees it.
    score = _run_label_validator(
        cfg, score, det.det_map, instruction, rgb, wrist_rgb_for_llm,
        pick_hint, place_hint, trial_meta=trial_meta,
        slot_key='validator_actions',
        picks=picks, places=places, pick_counts=pick_counts)
    _stash_bt(trial_meta, score, key='bt_yaml')

    # LIBERO-Dyn hook point: perturbation protocols inject scripted scene
    # changes after planning has bound its keypoints (POST_PLAN) or arm a
    # step-triggered injector (MID_APPROACH / POST_GRASP_PLAN).  Never
    # allowed to break a trial.
    if post_plan_hook is not None:
        try:
            post_plan_hook(env, score, det)
        except Exception as e:
            if cfg.verbose:
                print(f"[post_plan_hook] failed: {e}")

    _is_compound_run = (picks is not None and places is not None
                         and (len(picks) > 1 or (pick_counts
                              and any(c > 1 for c in pick_counts))))
    if (shadow_select_and_execute is not None and shadow_enabled()
            and not cfg.no_gemini and not _is_compound_run):
        # Shadow-sim best-of-N: sample K BTs, run each in env, keep the first
        # that yields env.check_success(). The env IS the metric in sim, so
        # the winning rollout already counts as the real execution - no
        # double-stepping required.
        k = shadow_k()
        candidates = _plan_diverse_candidates(
            cfg, instruction, det.det_map, rgb, wrist_rgb_for_llm,
            pick_hint, place_hint, k=k, picks=picks, places=places,
            pick_counts=pick_counts)
        # Always include the validated primary plan as candidate 0.
        if not candidates or candidates[0] is not score:
            candidates = [score] + [c for c in candidates if c is not score]
        # Validate each diversity sample's labels (snap-or-retry) using the
        # same det_map so the executor doesn't choke on hallucinated labels.
        validated = []
        for ci in candidates[:k]:
            ci_v = _run_label_validator(
                cfg, ci, det.det_map, instruction, rgb, wrist_rgb_for_llm,
                pick_hint, place_hint, trial_meta=None,
                slot_key='validator_actions',
                picks=picks, places=places, pick_counts=pick_counts)
            validated.append(ci_v)

        def _exec_one(env_, score_):
            _execute(env_, score_, det, instruction, cfg, sam3,
                      prompts=prompts)

        winner, diag = shadow_select_and_execute(
            env, validated, _exec_one, max_candidates=k, verbose=cfg.verbose)
        if cfg.verbose:
            print(f"[shadow-sim] K={len(validated)} "
                  f"rollouts={[r.get('success') for r in diag.get('rollouts', [])]} "
                  f"winner_idx={diag.get('winner_idx')}",
                  flush=True)
        if trial_meta is not None:
            trial_meta['shadow_sim'] = {
                'k': len(validated),
                'rollouts': diag.get('rollouts', []),
                'winner_idx': diag.get('winner_idx'),
            }
            # Persist EVERY candidate's full YAML pre-selection so losing
            # candidates stay available to the trace-selector's off-policy
            # pair corpus.
            try:
                trial_meta['shadow_candidates'] = [
                    yaml.safe_dump(
                        {k: v for k, v in c.items()
                         if not str(k).startswith('__')},
                        sort_keys=False)
                    for c in validated]
            except Exception:
                pass
        if winner is not None and winner is not score:
            score = winner
            _stash_bt(trial_meta, score, key='bt_yaml')
    else:
        _execute(env, score, det, instruction, cfg, sam3, prompts=prompts,
                  trial_meta=trial_meta)
    success = bool(env.check_success())

    # Lowest-responsible-layer recovery (ANCHOR pattern; opt-in via
    # cfg.layered_recovery): attribute the failure to perception /
    # execution / plan and repair at that layer instead of always
    # replanning.  Works under --no-gemini too (the plan rung is simply
    # unavailable there).
    if not success and getattr(cfg, 'layered_recovery', False):
        success = _layered_recovery(
            env, model, data, cfg, sam3, score, det, instruction, prompts,
            rgb, wrist_rgb_for_llm, pick_hint, place_hint,
            picks, places, pick_counts, trial_meta, _stash_bt)

    # Legacy blind recovery: retract + re-detect + re-plan + re-execute
    # (up to 2 tries).
    elif not success and not cfg.no_gemini:
        ee_site = _find_ee_site(model)
        for _ in range(2):
            _retract_arm(env, model, data, ee_site)
            # Re-check goal predicate AFTER retract. For push tasks the
            # plate may still be sliding when the post-execute check fires;
            # once the arm lifts and the sim settles the predicate often
            # flips True, and a further recovery push would knock it out.
            if _oracle_gate(env):
                success = True
                break
            # Privileged mode: re-fetch via GT, otherwise SAM3 re-detect.
            if getattr(cfg, 'privileged', False):
                obs2 = env._get_observations() if hasattr(env, '_get_observations') else {}
                det2 = detect_scene_privileged(env, prompts, obs2, cfg)
            else:
                det2 = redetect_agentview(env, sam3, prompts, cfg)
            if det2 is None:
                continue
            score2 = _plan(cfg, instruction, det2.det_map, det2.rgb,
                            wrist_rgb_for_llm, pick_hint, place_hint,
                            picks=picks, places=places,
                            pick_counts=pick_counts, prompts=prompts)
            if not score2:
                continue
            score2 = _run_label_validator(
                cfg, score2, det2.det_map, instruction, det2.rgb,
                wrist_rgb_for_llm, pick_hint, place_hint,
                trial_meta=trial_meta,
                slot_key='validator_actions_recovery',
                picks=picks, places=places, pick_counts=pick_counts)
            _stash_bt(trial_meta, score2, key='bt_yaml_recovery')
            _execute(env, score2, det2, instruction, cfg, sam3,
                      prompts=prompts, trial_meta=trial_meta)
            success = bool(env.check_success())
            if success:
                break

    if success and score is not None and not _bt_frozen():
        _store_in_bt_library(instruction, score, det.dets)

    if cfg.verbose:
        _debug_gt_dump(env, model, data)

    return success


def _debug_gt_dump(env, model, data) -> None:
    """
    DIAGNOSIS-ONLY end-of-trial ground-truth dump (verbose runs).

    Prints task-object body positions + articulated joint qpos so failure
    triage can compare perception/intent against where objects actually
    ended up.  Never feeds the pipeline (print-only, after success is
    already computed).
    """
    try:
        mujoco.mj_forward(model, data)
        print("[debug-gt] end-of-trial state:")
        for bid in range(model.nbody):
            bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
            low = bn.lower()
            if not low.endswith('_main') or 'robot' in low or 'gripper' in low:
                continue
            p = data.xpos[bid]
            print(f"[debug-gt]   {bn:45s} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        for jid in range(model.njnt):
            jt = int(model.jnt_type[jid])
            if jt not in (int(mujoco.mjtJoint.mjJNT_SLIDE),
                          int(mujoco.mjtJoint.mjJNT_HINGE)):
                continue
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ''
            bid = int(model.jnt_bodyid[jid])
            bn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                  or '').lower()
            if any(k in bn for k in ('robot', 'gripper', 'finger', 'link')):
                continue
            qpa = int(model.jnt_qposadr[jid])
            print(f"[debug-gt]   joint {jn:40s} qpos={data.qpos[qpa]:+.4f}")
    except Exception as e:  # pragma: no cover - diagnosis must never raise
        print(f"[debug-gt] dump failed: {e}")


# RGB-frame recorder (opt-in via --save-rgb-frames)

class _RGBFrameRecorder:
    """
    Monkey-patches ``env.step`` to collect per-step RGB + proprio tuples.

    Restored on ``stop()``.  Used downstream to train a V-JEPA adapter on the
    SPARK BT distribution - only the saved frames from *successful* trials
    are kept (see caller in ``run_suite``).
    """

    def __init__(self, env):
        self.env = env
        self._orig_step = env.step
        self.rgb_a_list: list = []
        self.rgb_w_list: list = []
        self.act_list: list = []
        self.ee_list: list = []  # 7-vec: pos(3) + quat(4)
        self.grip_list: list = []

        def _patched_step(action):
            ret = self._orig_step(action)
            try:
                # robosuite returns (obs, reward, done, info)
                obs = ret[0] if isinstance(ret, tuple) else ret
                if isinstance(obs, dict):
                    rgb_a = obs.get('agentview_image')
                    rgb_w = obs.get('robot0_eye_in_hand_image')
                    if rgb_a is not None and rgb_w is not None:
                        # Flip vertically (LIBERO renders upside down).
                        self.rgb_a_list.append(np.ascontiguousarray(
                            rgb_a[::-1], dtype=np.uint8))
                        self.rgb_w_list.append(np.ascontiguousarray(
                            rgb_w[::-1], dtype=np.uint8))
                        self.act_list.append(np.asarray(
                            action, dtype=np.float32).copy())
                        ee_pos = np.asarray(obs.get(
                            'robot0_eef_pos', np.zeros(3)), dtype=np.float32)
                        ee_quat = np.asarray(obs.get(
                            'robot0_eef_quat', np.zeros(4)), dtype=np.float32)
                        self.ee_list.append(np.concatenate([ee_pos, ee_quat]))
                        self.grip_list.append(np.asarray(obs.get(
                            'robot0_gripper_qpos', np.zeros(2)),
                            dtype=np.float32))
            except Exception:
                pass  # never block the trial because recording failed
            return ret

        env.step = _patched_step

    def stop(self) -> None:
        try:
            self.env.step = self._orig_step
        except Exception:
            pass

    def save(self, path: str, *, task_name: str, instruction: str) -> int:
        """
        Write buffers to HDF5.  Returns file size in bytes (0 on failure).
        """
        if not self.rgb_a_list or h5py is None:
            return 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, 'w') as f:
            f.create_dataset('agentview_rgb',
                              data=np.array(self.rgb_a_list, dtype=np.uint8),
                              compression='gzip', compression_opts=4)
            f.create_dataset('wrist_rgb',
                              data=np.array(self.rgb_w_list, dtype=np.uint8),
                              compression='gzip', compression_opts=4)
            f.create_dataset('actions',
                              data=np.array(self.act_list, dtype=np.float32))
            f.create_dataset('ee_poses',
                              data=np.array(self.ee_list, dtype=np.float32))
            f.create_dataset('grippers',
                              data=np.array(self.grip_list, dtype=np.float32))
            f.attrs['task_name'] = task_name
            f.attrs['instruction'] = instruction
            f.attrs['success'] = True
        try:
            return os.path.getsize(path)
        except OSError:
            return 0


# Suite runner

