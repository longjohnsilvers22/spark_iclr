"""
Frozen-VLA baselines evaluated under SPARK's own LIBERO-PRO protocol.

The point of this package is protocol parity, not policy engineering.  Every
runner here reproduces, trial for trial, the loop in
``spark_bench.run_spark_libero_pro_fair`` -- same benchmark registration, same
task ordering, same ``env.seed(trial)``, same init-state indexing, same settle,
same ``env.check_success()`` arbiter -- and writes the same JSON schema, so
``spark_bench.stats_intervals`` consumes SPARK arms and VLA arms unchanged.

Modules
-------
run_pi05_libero_pro
    Frozen pi0.5 (openpi ``pi05_libero``) on the six position/task cells.
"""
