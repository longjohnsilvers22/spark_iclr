"""
Tests for the frontend progress feed.

These cover the two defects that made a 133-second run render as one
silent block followed by a burst of identically-timestamped lines:

  1. The _ProgressHandler was attached to a hand-maintained allowlist of
     leaf logger names that had rotted. The executor's real loggers
     (spark_real.control.executor_*) were not on it, so the buffer stayed
     empty for the whole run even though /api/progress answered every
     poll with 200 OK.
  2. The client tracked its position with an index into the returned
     list, which desynchronises the moment the capped buffer trims.
"""

import asyncio
import logging

import pytest

from spark_real.routes import core, state


@pytest.fixture(autouse=True)
def _clean_progress():
    """
    Give every test an empty buffer and a known sequence origin.
    """
    with state.progress_lock:
        state.progress_log.clear()
        state.progress_seq = 0
    yield
    with state.progress_lock:
        state.progress_log.clear()
        state.progress_seq = 0


@pytest.fixture
def attached_root():
    """
    Exercise the handler server.py ACTUALLY installs.

    Importing spark_real.server runs its module-level attach loop, so the
    "spark_real" root already carries a _ProgressHandler. Adding another
    here would double every entry -- and the point of these tests is to
    verify the real wiring, not a replica of it. We only force the logger
    level, since pytest does not guarantee INFO is enabled.
    """
    import spark_real.server  # noqa: F401  (import for its attach side effect)

    root = logging.getLogger("spark_real")
    installed = [
        h for h in root.handlers if type(h).__name__ == "_ProgressHandler"
    ]
    assert len(installed) == 1, (
        "expected exactly one _ProgressHandler on the spark_real root, "
        f"found {len(installed)}"
    )
    prev = root.level
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.setLevel(prev)


# The modules that actually emit step messages during a run. Every one of
# these was invisible under the old allowlist. Regression guard: if the
# executor is refactored again, these must keep reaching the feed.
EXECUTOR_LOGGERS = [
    "spark_real.control.executor_core",
    "spark_real.control.executor_motion",
    "spark_real.control.executor_grasp",
    "spark_real.control.executor_release",
    "spark_real.control.executor_verify",
    "spark_real.control.primitive_timeouts",
    "spark_real.control.success_verifier",
    "spark_real.pipeline_execution",
    "spark_real.pipeline_perception",
    "spark_real.pipeline_io",
]


@pytest.mark.parametrize("name", EXECUTOR_LOGGERS)
def test_executor_loggers_reach_the_progress_buffer(attached_root, name):
    """
    A step message from any real executor/pipeline module lands in the feed.
    """
    logging.getLogger(name).info("[branch] move_to_keypoint")
    with state.progress_lock:
        msgs = [e["msg"] for e in state.progress_log]
    assert "[branch] move_to_keypoint" in msgs, (
        f"{name} did not reach the progress buffer; the handler attach "
        "point has rotted again"
    )


def test_score_executor_shim_emits_nothing():
    """
    The module the old allowlist named is a pure composition shim.

    It still imports, which is why the rot was invisible, but it holds no
    logger calls. Pinning this documents WHY naming it was useless.
    """
    from pathlib import Path

    import spark_real.control.score_executor as se

    src = Path(se.__file__).read_text()
    assert "logger." not in src


def test_progress_entries_carry_event_time_and_sequence(attached_root):
    """
    Each entry gets a monotonic seq and the record's own creation time.
    """
    log = logging.getLogger("spark_real.control.executor_core")
    log.info("first")
    log.info("second")
    with state.progress_lock:
        entries = list(state.progress_log)
    assert [e["msg"] for e in entries] == ["first", "second"]
    assert [e["seq"] for e in entries] == [1, 2]
    # Real clock values, and non-decreasing.
    assert entries[0]["ts"] > 0
    assert entries[1]["ts"] >= entries[0]["ts"]


def test_since_returns_only_newer_entries(attached_root):
    """
    A cursor poll gets strictly what followed it, and a fresh next_since.
    """
    log = logging.getLogger("spark_real.control.executor_core")
    log.info("a")
    log.info("b")

    first = asyncio.run(core.get_progress(since=0))
    assert [e["msg"] for e in first["log"]] == ["a", "b"]
    cursor = first["next_since"]

    # Nothing new yet.
    idle = asyncio.run(core.get_progress(since=cursor))
    assert idle["log"] == []
    assert idle["next_since"] == cursor

    log.info("c")
    nxt = asyncio.run(core.get_progress(since=cursor))
    assert [e["msg"] for e in nxt["log"]] == ["c"]


def test_omitting_since_returns_whole_buffer(attached_root):
    """
    Back-compat: an old frontend that sends no cursor still gets the log.
    """
    logging.getLogger("spark_real.control.executor_core").info("x")
    res = asyncio.run(core.get_progress())
    assert [e["msg"] for e in res["log"]] == ["x"]


def test_cursor_survives_buffer_trim(attached_root, monkeypatch):
    """
    The bug an index-based cursor had: trimming must not skip or repeat.

    Overflow the cap by more than the client's read window and confirm
    the cursor still yields every surviving entry exactly once, in order.
    """
    monkeypatch.setattr(state, "PROGRESS_MAX", 10)
    log = logging.getLogger("spark_real.control.executor_core")

    seen = []
    cursor = 0
    for i in range(50):
        log.info("step-%d" % i)
        # Poll every 7 messages, so the buffer trims between reads.
        if i % 7 == 6:
            res = asyncio.run(core.get_progress(since=cursor))
            seen.extend(e["msg"] for e in res["log"])
            cursor = res["next_since"]
    res = asyncio.run(core.get_progress(since=cursor))
    seen.extend(e["msg"] for e in res["log"])

    # No duplicates, and strictly increasing order.
    assert len(seen) == len(set(seen))
    order = [int(m.split("-")[1]) for m in seen]
    assert order == sorted(order)
    # The buffer is capped, so early entries may legitimately be lost; but
    # everything from the last poll window onward must be present.
    assert "step-49" in seen


def test_buffer_respects_the_cap(attached_root, monkeypatch):
    monkeypatch.setattr(state, "PROGRESS_MAX", 5)
    log = logging.getLogger("spark_real.control.executor_core")
    for i in range(20):
        log.info("m%d" % i)
    with state.progress_lock:
        entries = list(state.progress_log)
    assert len(entries) == 5
    assert [e["msg"] for e in entries] == ["m%d" % i for i in range(15, 20)]


def test_clear_keeps_sequence_monotonic(attached_root):
    """
    DELETE must not reset seq, or a stale cursor would replay old lines.
    """
    log = logging.getLogger("spark_real.control.executor_core")
    log.info("before")
    cleared = asyncio.run(core.clear_progress())
    assert cleared["cleared"] is True
    after_clear = cleared["next_since"]
    assert after_clear >= 1

    log.info("after")
    res = asyncio.run(core.get_progress(since=after_clear))
    assert [e["msg"] for e in res["log"]] == ["after"]


def test_debug_records_are_filtered(attached_root):
    """
    The handler sits at INFO; debug chatter must not flood the operator.
    """
    logging.getLogger("spark_real.control.executor_core").debug("noisy")
    with state.progress_lock:
        assert state.progress_log == []


def test_end_to_end_over_http(attached_root):
    """
    The whole path the browser uses, through real HTTP.

    The other tests call the route function directly, which bypasses
    FastAPI's query-string parsing -- so this is what actually proves
    `?since=` is wired up rather than silently ignored.
    """
    from fastapi.testclient import TestClient

    import spark_real.server as srv

    client = TestClient(srv.app)

    logging.getLogger("spark_real.control.executor_core").info(
        "[branch] move_to_keypoint"
    )
    first = client.get("/api/progress")
    assert first.status_code == 200
    body = first.json()
    assert [e["msg"] for e in body["log"]] == ["[branch] move_to_keypoint"]
    cursor = body["next_since"]

    logging.getLogger("spark_real.control.executor_grasp").info("[2/3] grasp")
    second = client.get("/api/progress", params={"since": cursor})
    assert second.status_code == 200
    nxt = second.json()
    # Strictly the new line -- the cursor was honoured, not ignored.
    assert [e["msg"] for e in nxt["log"]] == ["[2/3] grasp"]
    assert nxt["next_since"] == cursor + 1
