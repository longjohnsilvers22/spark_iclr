"""
Per-primitive handlers for ``LiberoExecutor``.

Each module exposes a ``handle(executor, params)`` function (and any
private helpers it needs).  ``executor.py`` imports them all into a
single ``HANDLERS`` registry indexed by the BT primitive ``type``.
"""
