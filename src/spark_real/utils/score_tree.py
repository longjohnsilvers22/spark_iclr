"""Walk and validate the dict tree of a SPARK score."""

from spark_real.bt_label_resolver import base_label


def is_num(value) -> bool:
    # bool is an int subclass; a bool where a number belongs is a schema error.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def walk_nodes(node, path: str = "tree"):
    """Yield (path, node) for every dict node in a tree, depth first."""
    if not isinstance(node, dict):
        return
    yield path, node
    for i, child in enumerate(node.get("children") or []):
        yield from walk_nodes(child, f"{path}/{i}")


def label_known(label: str, known) -> bool:
    """True if `label` names one of the detected keypoints.

    Tolerates instance numbering in either direction ("fork 1" vs "fork"),
    matching how LabelResolvingDetectionMap binds labels at execution time.
    """
    if not known:
        return True
    if label in known:
        return True
    base = base_label(label)
    return base in known or any(base_label(k) == base for k in known)
