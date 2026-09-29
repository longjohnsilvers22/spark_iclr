"""
FastAPI service that wraps SAM3 so multiple workers can share one GPU
copy of the model. Mirrors CaP-X's ``capx/serving/launch_sam3_server.py``
architecture.

Server: ``python -m sam3_service.server`` (default port 8115).
Client: ``spark_bench.libero_pro.sam3_client`` exposes a thin wrapper that
matches the in-process ``Sam3Processor`` API so existing perception code
can opt-in via ``cfg.use_sam3_service=True``.
"""
