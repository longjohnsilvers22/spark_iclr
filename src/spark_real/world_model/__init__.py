"""V-JEPA 2-AC BT verifier for SPARK.

Zero-training: load frozen Meta V-JEPA 2-AC checkpoint, encode the current
observation and a goal frame, roll candidate BTs (compiled to 7-DoF EE
deltas) through the predictor in latent space, and rank by L2-to-goal.

Modules:
- vjepa2_ac : thin loader and encode/predict wrappers
- bt_to_actions : BT -> 7-DoF EE delta sequence
- bt_verifier : main K-candidate ranking loop
"""

from .vjepa2_ac import load_vjepa2_ac, encode_frame, predict_next_latent

__all__ = ["load_vjepa2_ac", "encode_frame", "predict_next_latent"]
