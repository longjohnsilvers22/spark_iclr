#!/usr/bin/env python3
"""
SPARK on LIBERO-PRO: Fair evaluation matching CaP-X protocol.

Uses LIBERO's actual ``OffScreenRenderEnv`` with:

* Proper init states from ``.pruned_init`` files (50 per task).
* LIBERO's canonical ``check_success()`` (BDDL predicate evaluation).
* Same perturbation types (position swap, task change).
* Multiple trials per task with different init states.

Runs in the openvla conda env (has LIBERO + robosuite + SAM3).

Usage::

    conda activate openvla
    MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src/sam3:src:$PYTHONPATH \\
        python -m spark_bench.run_spark_libero_pro_fair \\
        --suite object --perturbation position --num-trials 5

This module is a thin orchestration shell - BDDL parsing, planning,
perception, IK, motion, and BT execution all live in
``spark_bench.libero_pro.{bddl, planning, perception, ik, motion,
executor}``.  See ``libero_pro/__init__.py`` for the package map.
"""
from __future__ import annotations
from .fair.config import (
    FairConfig,
    Optional,
    Path,
    _gemini_plan,
    _get_sam3,
    _load_init_states_fallback,
    _scripted_plan,
    benchmark,
    execute_on_libero,
    get_libero_path,
    get_perturbation_suite,
    get_task_prompts_for_suite,
    json,
    load_libero_env,
    np,
    os,
    run_atomic_decomp,
    tyro,
)
from .fair.execution import (
    _RGBFrameRecorder,
    run_spark_on_libero_env,
)


def run_suite(cfg: FairConfig):
    """
    Run LIBERO-PRO evaluation for one suite.
    """
    base_suite = f"libero_{cfg.suite}"
    perturbation_types = ([cfg.perturbation] if cfg.perturbation != 'all'
                           else ['position', 'task'])

    benchmark_dict = benchmark.get_benchmark_dict()
    all_results: dict = {}

    # Resume: a partial checkpoint from a previous process pre-fills
    # completed (perturbation, task) cells so a reboot costs minutes, not a
    # suite. Written by the per-task checkpoint below. 2026-09-02: a host
    # reboot ate three 50-trial suites at ~80 percent for want of this.
    _resume = {}
    try:
        _ck_dir = (Path(cfg.output_dir) if cfg.output_dir
                    else Path.home() / 'spark' / 'videos' / 'libero_pro_fair')
        _ckf = _ck_dir / f'{cfg.suite}.partial.json'
        if _ckf.exists():
            with open(_ckf) as _f:
                _prev = json.load(_f)
            if _prev.get('num_trials') == cfg.num_trials:
                _resume = _prev.get('perturbations', {})
                print('[resume] checkpoint found:',
                      {k: len(v.get('task_details', []))
                       for k, v in _resume.items()}, 'tasks banked')
    except Exception as _e:  # noqa: BLE001
        print(f'[resume] checkpoint unreadable, fresh start: {_e}')
        _resume = {}

    for ptype in perturbation_types:
        pert_suite = get_perturbation_suite(base_suite, ptype)
        if pert_suite not in benchmark_dict:
            print(f"Note: {pert_suite} not registered, "
                  f"using base suite {base_suite}")
            pert_suite = base_suite

        task_suite = benchmark_dict[pert_suite]()
        num_tasks = task_suite.n_tasks

        print(f"\n{'='*60}")
        print(f"LIBERO-PRO Fair: {cfg.suite} suite, perturbation={ptype}")
        print(f"Tasks: {num_tasks}, Trials per task: {cfg.num_trials}")
        print(f"{'='*60}\n")
        if cfg.save_rgb_frames:
            # ~256*256*3*2*200/1e6 ~ 80 MB per successful trial.  Multiply
            # by an upper-bound on successful trials per cell (== num_trials)
            # to give the operator a worst-case footprint estimate.  Uses
            # the actual configured cam_width/cam_height.
            est_mb_per_trial = (
                cfg.cam_width * cfg.cam_height * 3 * 2 * 200 / 1e6)
            est_mb = est_mb_per_trial * cfg.num_trials * num_tasks
            print(f"[fair] save_rgb_frames=True; expect "
                   f"~{est_mb_per_trial:.0f}MB per successful trial "
                   f"(~{est_mb/1024:.1f}GB worst-case for this suite)")

        task_results: list[float] = []
        task_details: list[dict] = []

        skip_ids = {int(s) for s in cfg.skip_task_ids.split(',')
                     if s.strip().isdigit()}
        for task_id in range(num_tasks):
            _banked = _resume.get(ptype, {}).get('task_details', [])
            if task_id < len(_banked):
                _d = _banked[task_id]
                task_results.append(float(_d.get('success_rate', 0.0)))
                task_details.append(_d)
                print(f'[resume] {ptype} task {task_id} banked '
                      f"({_d.get('success_rate', 0.0):.0%}), skipping")
                continue
            if cfg.only_task_id >= 0 and task_id != cfg.only_task_id:
                continue
            if task_id < cfg.start_task_id:
                continue
            if cfg.end_task_id >= 0 and task_id >= cfg.end_task_id:
                continue
            if task_id in skip_ids:
                print(f"[skip] task_id={task_id} (in skip_task_ids)")
                continue
            task = task_suite.get_task(task_id)
            try:
                init_states = task_suite.get_task_init_states(task_id)
            except Exception:
                init_states = _load_init_states_fallback(pert_suite, task)
                if init_states is None and task_id == 0:
                    print(f"WARNING: No init states for {pert_suite} - "
                          f"using default reset")
                elif init_states is not None and task_id == 0:
                    print(f"[init_states fallback] {pert_suite} -> "
                          f"{len(init_states)} states from "
                          f"libero_pro/init_files")

            bddl_path = task_suite.get_task_bddl_file_path(task_id)
            if not os.path.exists(bddl_path):
                base_folder = '_'.join(task.problem_folder.split('_')[:2])
                bddl_path = os.path.join(get_libero_path("bddl_files"),
                                          base_folder, task.bddl_file)

            task_info = get_task_prompts_for_suite(base_suite, task_id, task,
                                                     bddl_path=bddl_path)
            task_name = task.name[:50]
            print(f"{task_name:50s}", end=" ", flush=True)

            trial_successes: list[bool] = []
            trial_metas: list[dict] = []
            env = None
            try:
                env, _, _, _ = load_libero_env(pert_suite, task_id, cfg)
            except Exception as e:
                print(f"ENV LOAD FAILED: {e}")
                task_results.append(0)
                task_details.append({
                    'task_name': task.name,
                    'instruction': task_info['instruction'],
                    'prompts': task_info['prompts'],
                    'pick': task_info.get('pick', ''),
                    'place': task_info.get('place', ''),
                    'success_rate': 0,
                    'error': str(e),
                })
                continue

            for trial in range(cfg.num_trials):
                trial_meta: dict = {'trial': trial}
                recorder: Optional[_RGBFrameRecorder] = None
                try:
                    env.seed(trial + getattr(cfg, 'trial_offset', 0))
                    env.reset()
                    if init_states is not None and len(init_states) > 0:
                        state_idx = ((trial + getattr(cfg, 'trial_offset', 0))
                                      % len(init_states))
                        env.set_init_state(init_states[state_idx])
                        env.reset()
                    for _ in range(10):
                        env.step(np.zeros(7))

                    # Start recording AFTER the 10-step settle so the saved
                    # stream only contains pipeline-controlled frames.
                    if cfg.save_rgb_frames:
                        recorder = _RGBFrameRecorder(env)

                    if getattr(cfg, 'atomic_decomp', False):
                        success = run_atomic_decomp(
                            env, task_info['prompts'],
                            task_info['instruction'], cfg, bddl_path)
                    else:
                        success = run_spark_on_libero_env(
                            env, task_info['prompts'],
                            task_info['instruction'],
                            task_info.get('pick', ''),
                            task_info.get('place', ''), cfg,
                            trial_meta=trial_meta,
                            picks=task_info.get('picks') or None,
                            places=task_info.get('places') or None,
                            pick_counts=task_info.get('pick_counts') or None)
                    trial_successes.append(success)
                    trial_meta['success'] = bool(success)
                except Exception as e:
                    if cfg.verbose:
                        print(f"\n    Trial {trial} error: {e}")
                    trial_successes.append(False)
                    trial_meta['success'] = False
                    trial_meta['error'] = str(e)
                finally:
                    if recorder is not None:
                        recorder.stop()
                        if trial_meta.get('success'):
                            out_dir = (Path(cfg.output_dir) if cfg.output_dir
                                        else Path.home() / 'spark' / 'videos'
                                        / 'libero_pro_fair')
                            rgb_path = (out_dir / 'rgb_streams'
                                        / f'{task.name}_T{trial}.h5')
                            try:
                                nbytes = recorder.save(
                                    str(rgb_path),
                                    task_name=task.name,
                                    instruction=task_info['instruction'])
                                trial_meta['rgb_stream_path'] = str(rgb_path)
                                trial_meta['rgb_stream_bytes'] = nbytes
                                if cfg.verbose:
                                    print(f"\n    [save_rgb_frames] wrote "
                                           f"{rgb_path} ({nbytes/1e6:.1f}MB)")
                            except Exception as e:
                                if cfg.verbose:
                                    print(f"\n    [save_rgb_frames] failed: {e}")
                trial_metas.append(trial_meta)

                # Live progress: tick per trial.
                _n = len(trial_successes)
                _wins = sum(trial_successes)
                print("*" if not trial_successes[-1] else "ok", end="", flush=True)
                if _n % 10 == 0:
                    print(f"[{_wins}/{_n}]", end="", flush=True)

            try:
                env.close()
            except Exception:
                pass

            task_sr = (sum(trial_successes) / len(trial_successes)
                        if trial_successes else 0)
            task_results.append(task_sr)
            task_details.append({
                'task_name': task.name,
                'instruction': task_info['instruction'],
                'prompts': task_info['prompts'],
                'pick': task_info.get('pick', ''),
                'place': task_info.get('place', ''),
                'success_rate': task_sr,
                'trials': [bool(s) for s in trial_successes],
                'trial_meta': trial_metas,
            })
            print(f"{task_sr:.0%} ({sum(trial_successes)}/{len(trial_successes)})")

            # Checkpoint after EVERY task; atomic replace so a mid-write
            # reboot cannot corrupt the file.
            try:
                _ck_dir.mkdir(parents=True, exist_ok=True)
                _partial = dict(all_results)
                _partial[ptype] = {
                    'per_task': list(task_results),
                    'average': (float(np.mean(task_results))
                                 if task_results else 0.0),
                    'task_details': list(task_details),
                }
                _tmp = _ck_dir / f'{cfg.suite}.partial.json.tmp'
                with open(_tmp, 'w') as _f:
                    json.dump({'suite': cfg.suite,
                               'num_trials': cfg.num_trials,
                               'perturbations': _partial}, _f, indent=2)
                os.replace(_tmp, _ck_dir / f'{cfg.suite}.partial.json')
            except Exception as _e:  # noqa: BLE001
                print(f'[checkpoint] write failed: {_e}')

        avg_sr = float(np.mean(task_results)) if task_results else 0.0
        print(f"\n{cfg.suite} ({ptype}): {avg_sr:.1%}")
        all_results[ptype] = {
            'per_task': task_results,
            'average': avg_sr,
            'task_details': task_details,
        }

    print(f"\n{'='*60}")
    print(f"LIBERO-PRO Fair Summary ({cfg.suite})")
    print(f"{'='*60}")
    total_sr: list[float] = []
    for ptype, res in all_results.items():
        print(f"{ptype:12s}: {res['average']:.1%}")
        total_sr.append(res['average'])
    if total_sr:
        print(f"{'Overall':12s}: {np.mean(total_sr):.1%}")

    out_dir = (Path(cfg.output_dir) if cfg.output_dir
                else Path.home() / 'spark' / 'videos' / 'libero_pro_fair')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{cfg.suite}.json', 'w') as f:
        json.dump({
            'suite': cfg.suite,
            'num_trials': cfg.num_trials,
            'perturbations': all_results,
        }, f, indent=2)


# Backwards-compat shim: older callers may still import _execute_on_libero.

_execute_on_libero = execute_on_libero


def main():
    cfg = tyro.cli(FairConfig)
    run_suite(cfg)


if __name__ == '__main__':
    main()
