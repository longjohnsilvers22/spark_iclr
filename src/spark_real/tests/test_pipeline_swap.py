"""
/api/initialize must not swap a live pipeline out from under its devices.

The handler did::

    if state.pipeline is not None and state.pipeline._initialized:
        return already_initialized
    ...
    state.pipeline = SPARKRealPipeline(config, profile=profile)
    state.pipeline.initialize()

Two defects in three lines.

1. Reaching the assignment means a pipeline EXISTS whose initialize() did not
   finish -- a boot that raised, or a previous /api/initialize that failed
   halfway. That object still owns whatever it did open: PyK4A handles with
   live capture threads, the RealSense stream, the RTDE connection.
   Overwriting the reference just dropped them, so the new initialize() ran
   PyK4A.start() on devices the old object was still streaming from -- two
   handles on one depth MCU, with the old capture threads still calling
   get_capture(). Nothing released them afterwards either: atexit's
   _release_devices walks state.pipeline, which by then pointed elsewhere.

2. state.pipeline was published BEFORE initialize() ran, so every concurrent
   reader -- the stream pool, /api/status, the recorder -- saw a
   half-constructed object for the whole multi-second SAM3 load; and an
   initialize() that raised left that wreck installed as the live pipeline.

No hardware; SPARKRealPipeline is replaced by a recording fake.
"""

import asyncio
import threading

import pytest

from spark_real.routes import core as core_mod
from spark_real.routes import state


@pytest.fixture(autouse=True)
def _clean_pipeline(monkeypatch):
    monkeypatch.setattr(state, "pipeline", None)
    monkeypatch.setattr(state, "launch_config", None)
    # A fresh lock per test; the module-level one is bound to another loop.
    monkeypatch.setattr(state, "pipeline_lock", asyncio.Lock())
    yield


def _install_fake(monkeypatch, events, *, fails=False, on_init=None):
    class _Fake:
        instances = []

        def __init__(self, config, profile=None):
            self.config = config
            self._initialized = False
            self.shutdown_calls = 0
            self.index = len(_Fake.instances)
            _Fake.instances.append(self)
            events.append(f"construct#{self.index}")

        def initialize(self):
            events.append(f"initialize#{self.index}")
            if on_init is not None:
                on_init(self)
            if fails:
                raise RuntimeError("camera bring-up failed")
            self._initialized = True

        def shutdown(self):
            self.shutdown_calls += 1
            events.append(f"shutdown#{self.index}")
            self._initialized = False

        def get_status(self):
            assert self._initialized, (
                "get_status() reached a pipeline that never finished "
                "initialize()"
            )
            return {"ok": True}

    monkeypatch.setattr(core_mod, "SPARKRealPipeline", _Fake)
    monkeypatch.setattr(core_mod, "PipelineConfig", lambda **kw: dict(kw))
    return _Fake


def test_a_half_built_pipeline_is_released_before_a_new_one_is_built(monkeypatch):
    events = []
    fake = _install_fake(monkeypatch, events)

    # A previous boot got partway and left its devices held.
    stale = object.__new__(fake)
    stale._initialized = False
    stale.shutdown_calls = 0
    stale.index = "stale"
    stale.shutdown = lambda: events.append("shutdown#stale")
    monkeypatch.setattr(state, "pipeline", stale)

    asyncio.run(core_mod.initialize_pipeline())

    assert events.index("shutdown#stale") < events.index("construct#0"), (
        "the new pipeline was constructed before the old one released its "
        "Kinect handles"
    )
    assert events.index("shutdown#stale") < events.index("initialize#0")


def test_an_already_initialized_pipeline_is_left_alone(monkeypatch):
    events = []
    _install_fake(monkeypatch, events)

    class _Live:
        _initialized = True

        def get_status(self):
            return {"ok": True}

    live = _Live()
    monkeypatch.setattr(state, "pipeline", live)
    out = asyncio.run(core_mod.initialize_pipeline())
    assert out["status"] == "already_initialized"
    assert events == []
    assert state.pipeline is live


def test_pipeline_is_published_only_after_initialize_returns(monkeypatch):
    """
    A concurrent reader must never see a pipeline that is still bringing its
    cameras up.
    """
    events = []
    observed = {}

    def _on_init(inst):
        # Runs during initialize(), i.e. exactly when a stream-pool worker or
        # /api/status would be dereferencing state.pipeline.
        observed["published_during_init"] = state.pipeline

    _install_fake(monkeypatch, events, on_init=_on_init)
    asyncio.run(core_mod.initialize_pipeline())

    assert observed["published_during_init"] is None, (
        "a half-constructed pipeline was visible to concurrent readers for "
        "the whole of initialize()"
    )
    assert state.pipeline is not None
    assert state.pipeline._initialized is True


def test_a_failed_initialize_is_not_left_installed_and_releases_devices(monkeypatch):
    events = []
    fake = _install_fake(monkeypatch, events, fails=True)

    with pytest.raises(RuntimeError, match="camera bring-up failed"):
        asyncio.run(core_mod.initialize_pipeline())

    assert state.pipeline is None, (
        "a pipeline whose initialize() raised was left installed as the live "
        "pipeline"
    )
    assert fake.instances[0].shutdown_calls == 1, (
        "the failed pipeline kept whatever devices it managed to open"
    )
    assert events == ["construct#0", "initialize#0", "shutdown#0"]


def test_repeated_failed_initializes_never_stack_open_devices(monkeypatch):
    """
    The operator hitting Initialize three times on a flaky rig must not end
    up with three PyK4A handles on one device.
    """
    events = []
    fake = _install_fake(monkeypatch, events, fails=True)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            asyncio.run(core_mod.initialize_pipeline())
    assert len(fake.instances) == 3
    assert [i.shutdown_calls for i in fake.instances] == [1, 1, 1]
    assert state.pipeline is None


def test_concurrent_initializes_build_exactly_one_pipeline(monkeypatch):
    """The asyncio lock has to actually serialise the build."""
    events = []
    fake = _install_fake(monkeypatch, events)
    gate = threading.Event()

    async def _main():
        return await asyncio.gather(
            core_mod.initialize_pipeline(),
            core_mod.initialize_pipeline(),
            core_mod.initialize_pipeline(),
        )

    asyncio.run(_main())
    gate.set()
    built = [e for e in events if e.startswith("construct")]
    assert len(built) == 1, events
    assert state.pipeline._initialized is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
