"""Field access on a detection given as a dict OR a dataclass."""


def det_field(det, key, default=None):
    """Read `key` from `det`; None (missing det or missing/None value) -> default."""
    if det is None:
        return default
    v = det.get(key, default) if hasattr(det, "get") else getattr(det, key, default)
    return default if v is None else v
