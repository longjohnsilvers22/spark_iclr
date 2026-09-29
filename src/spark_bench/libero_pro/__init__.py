"""
Components of the LIBERO-PRO fair runner
(``spark_bench.run_spark_libero_pro_fair``):

* ``ik.py``         - pure IK helpers (3-DOF, 6-DOF, pyroki sideways).
* ``motion.py``     - joint-space PD ``joint_move``, OSC ``move_to``,
                       gripper helpers, and MuJoCo discovery functions.
* ``executor.py``   - ``LiberoExecutor`` class with state + per-primitive
                       handlers; calls into ``ik`` / ``motion`` instead of
                       inlining everything.
* ``perception.py`` - prompt selection (tuned/Gemini/adaptive),
                       dual-camera SAM3 detection + agent/wrist merge,
                       redetection helpers for the recovery loop.
* ``planning.py``   - Gemini, scripted fallback, DSL-typed planner.
* ``bddl.py``       - BDDL parsing -> SAM3 prompts.
"""
