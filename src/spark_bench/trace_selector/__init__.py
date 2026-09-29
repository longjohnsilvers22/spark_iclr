"""
Predicate-trace selection model: a plan-success model mined from the
LLM-free verifier's own P0-a trial traces (no foundation-model fine-tuning).

Modules:
    features       - pre-execution feature extraction from (BT YAML, det context)
    mine           - flatten P0-a trial JSONs into a (trial, plan) dataset
    train          - logistic regression + GBT with task-level CV (needs sklearn)
    select         - deployable CPU ranker: rank_plans(candidates, det_map)
    eval_selection - off-policy selection accuracy on shadow-arm logs
"""
