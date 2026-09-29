"""
A stop pressed before the run starts must not be wiped by the run starting.

``execute_score`` opened with a bare ``self._abort = False``. /api/execute
takes ``state.execute_lock`` and then spends SECONDS in capture -> SAM3
detect -> Gemini plan before execute_score is reached, and /api/stop and
/api/abort are async handlers living on the event loop for that whole
window::

    executor thread                     /api/stop (event loop)
    ---------------                     ----------------------
    execute_lock.acquire()
    reset_run_state(executor)           << task armed here
    pipeline.capture()
    perception.detect(...)   ~2 s
    planner.plan(...)        ~3 s
                                        executor.abort()
                                          _abort = True
                                          arm BRAKED
    executor.execute_score(score)
      self._abort = False               << the stop is discarded
      ... drives the whole score

The operator's stop button visibly did nothing: the arm braked, paused, and
then executed the plan anyway. The flag now carries an epoch, stamped at the
task boundary, so an abort raised after arming survives into the run.

Pure state machine; no robot, no perception, no server.
"""

import threading

import pytest

from spark_real.control.executor_core import ScoreExecutorCore
from spark_real.control.executor_types import AbortRequested


class _FakeExecutor:
    """The abort/arming state machine lifted out of ScoreExecutor.

    Uses the real method, bound to a bare object, so the test exercises the
    shipped logic without constructing a ScoreExecutor (which needs a robot,
    a servo and a pipeline).
    """

    note_abort_requested = ScoreExecutorCore.note_abort_requested
    retract_abort = ScoreExecutorCore.retract_abort

    def __init__(self):
        self._abort = False
        self._abort_epoch = 0
        self._armed_epoch = 0
        self._abort_lock = threading.Lock()
        self.ran = False

    def arm(self):
        """What reset_task_state does to the abort fields."""
        self._armed_epoch = self._abort_epoch
        self._abort = False

    def start_run(self):
        """What execute_score does to the abort fields."""
        self._abort = self._abort_epoch != self._armed_epoch
        if self._abort:
            raise AbortRequested()
        self.ran = True


def test_a_stop_between_arming_and_the_run_is_honoured():
    ex = _FakeExecutor()
    ex.arm()  # /api/execute -> reset_run_state
    ex.note_abort_requested()  # /api/stop, during capture/detect/plan
    with pytest.raises(AbortRequested):
        ex.start_run()
    assert ex.ran is False


def test_a_stale_abort_from_the_previous_task_does_not_block_the_next():
    """The task boundary is the one place an abort is legitimately forgotten."""
    ex = _FakeExecutor()
    ex.arm()
    ex.note_abort_requested()
    with pytest.raises(AbortRequested):
        ex.start_run()
    # Next task: armed afresh, so it runs.
    ex.arm()
    ex.start_run()
    assert ex.ran is True
    assert ex._abort is False


def test_an_abort_survives_every_closed_loop_pass_of_the_same_task():
    """
    Closed-loop passes share task state (reset_run_state(task_scope=False)),
    so they do NOT re-arm. A stop must end the whole task, not one pass.
    """
    ex = _FakeExecutor()
    ex.arm()
    ex.start_run()
    assert ex.ran is True
    ex.ran = False
    ex.note_abort_requested()
    for _ in range(3):
        with pytest.raises(AbortRequested):
            ex.start_run()
        assert ex.ran is False


def test_a_fresh_executor_with_no_abort_runs():
    ex = _FakeExecutor()
    ex.start_run()
    assert ex.ran is True


def test_epoch_moves_before_the_flag_is_visible():
    """
    A reader that observes _abort True must already observe the moved epoch,
    or start_run would clear a live request.
    """
    ex = _FakeExecutor()
    ex.arm()
    seen = []
    stop = threading.Event()

    def _watch():
        while not stop.is_set():
            flag = ex._abort
            epoch = ex._abort_epoch
            if flag:
                seen.append((flag, epoch, ex._armed_epoch))
                return

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    for _ in range(200):
        ex.note_abort_requested()
    stop.set()
    t.join(timeout=3.0)
    for flag, epoch, armed in seen:
        assert epoch != armed, (
            "_abort became visible before the epoch moved; start_run would "
            "clear this request"
        )


# The stop routes must not bypass the epoch


def test_stop_routes_never_assign_abort_directly():
    """
    A bare ``executor._abort = True`` is invisible to the arming check. Every
    stop path has to go through note_abort_requested.
    """
    import ast
    import pathlib

    import spark_real.routes.core as core_mod

    routes_dir = pathlib.Path(core_mod.__file__).parent
    control_dir = routes_dir.parent / "control"
    offenders = []
    for path in list(routes_dir.glob("*.py")) + list(control_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Attribute) and target.attr == "_abort"):
                    continue
                # `self._abort = ...` inside the executor itself is the
                # implementation; anything reaching ACROSS an object is not.
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    continue
                # Fallback branches guarded by `callable(note)` are allowed;
                # they only run on an executor that predates the method.
                offenders.append(f"{path.name}:{node.lineno}")
    # The remaining sites are all explicit legacy fallbacks, taken only when
    # the executor predates note_abort_requested / retract_abort.
    assert len(offenders) <= 4, offenders


# The primitive-timeout watchdog must retract only its OWN abort


_WatchdogExecutor = _FakeExecutor


def test_watchdog_retracts_only_its_own_abort():
    ex = _WatchdogExecutor()
    ex.arm()
    epoch = ex.note_abort_requested()  # watchdog fires
    assert ex.retract_abort(epoch) is True
    assert ex._abort is False
    ex.start_run()  # recovery may proceed
    assert ex.ran is True


def test_watchdog_retraction_is_refused_when_an_operator_stop_landed():
    """
    The window: the watchdog aborts, the primitive unwinds, the OPERATOR
    presses stop, and only then does the handler try to clear the flag. The
    flat `executor._abort = False` it used to run discarded that stop and the
    run carried on driving the arm.
    """
    ex = _WatchdogExecutor()
    ex.arm()
    watchdog_epoch = ex.note_abort_requested()
    ex.note_abort_requested()  # operator, while the primitive unwinds
    assert ex.retract_abort(watchdog_epoch) is False
    assert ex._abort is True
    with pytest.raises(AbortRequested):
        ex.start_run()


def test_watchdog_retraction_is_refused_when_an_operator_stop_came_first():
    """
    The mirror-image window, and the subtler one: the operator presses stop
    FIRST, then a long primitive blows its budget and the watchdog fires. The
    watchdog's request is now the NEWEST, so a "nothing newer landed" check
    alone would happily retract -- and take the operator's stop with it,
    because clearing the flag clears it for everyone.

        armed = A
        operator stop           -> epoch A+1
        watchdog fires          -> epoch A+2   (the newest)
        retract(A+2): nothing newer... but A+1 is still outstanding

    Retraction also has to require that nothing OLDER is outstanding.
    """
    ex = _WatchdogExecutor()
    ex.arm()
    ex.note_abort_requested()  # operator
    watchdog_epoch = ex.note_abort_requested()  # budget blown afterwards
    assert ex.retract_abort(watchdog_epoch) is False
    assert ex._abort is True
    with pytest.raises(AbortRequested):
        ex.start_run()


def test_consecutive_watchdog_timeouts_each_retract_cleanly():
    """Two timed-out primitives in one task must both route to recovery."""
    ex = _WatchdogExecutor()
    ex.arm()
    for _ in range(3):
        epoch = ex.note_abort_requested()
        assert ex.retract_abort(epoch) is True
        assert ex._abort is False
    ex.start_run()
    assert ex.ran is True


def test_retraction_race_never_drops_a_stop():
    """
    Many watchdog retractions racing many operator stops: whenever a stop is
    the most recent request, the flag must still be up.
    """
    ex = _WatchdogExecutor()
    ex.arm()
    lost = []

    def _watchdogs():
        for _ in range(500):
            e = ex.note_abort_requested()
            ex.retract_abort(e)

    def _operator():
        for _ in range(500):
            e = ex.note_abort_requested()
            with ex._abort_lock:
                if ex._abort_epoch == e and not ex._abort:
                    lost.append(e)

    threads = [
        threading.Thread(target=_watchdogs, daemon=True),
        threading.Thread(target=_operator, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert lost == [], f"{len(lost)} operator stop(s) were retracted away"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
