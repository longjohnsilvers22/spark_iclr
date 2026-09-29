"""
Lightweight inter-process / inter-arm communication helpers.

Currently used by:

* :mod:`bimanual_zmq` is a ZMQ pub/sub bridge for high-rate dual-arm
  state broadcast.

The single-arm pipeline does not import anything from this package;
it stays Python-only. Bimanual code imports lazily so missing
``pyzmq`` degrades gracefully.
"""
