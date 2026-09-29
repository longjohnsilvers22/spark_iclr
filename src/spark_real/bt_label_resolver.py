"""
Keypoint-label resolution for cached behaviour trees.

A cached BT is written once and replayed forever, but the detections it
addresses are produced fresh by SAM3 on every run. The label a tree says
and the label the scene produces drift apart in four routine ways:

  1. **Instance arity.** One knife in frame is labelled ``knife handle``;
     two are labelled ``knife handle 1`` / ``knife handle 2``. A tree cached
     in a one-object scene hard-fails in a two-object scene, and vice versa.
  2. **Spelling.** A tree captured from a run that said ``grey block``
     cannot address a detection SAM3 now labels ``gray block``.
  3. **Plurals.** ``spoons 1`` vs ``spoon 1``.
  4. **Sub-part prompts.** A tree may say ``knife`` where the prompt
     registry now grounds ``knife handle`` (or the reverse).

Every consumer of the detection map does a bare ``detection_map.get(label)``
and hard-fails on a miss (``skills/primitives.py``,
``control/executor_motion.py``). Rather than patch each call site, the
pipeline hands the executor a dict subclass whose lookups fall back through
the drift cases above. Exact hits are untouched and cost nothing extra, so
behaviour for a freshly-planned tree is unchanged.

Resolution order for a miss, first hit wins:

    exact -> canonical spelling/plural -> base label (drop " <N>")
          -> sole instance of that base -> lowest-numbered instance
          -> unique sub-part / super-part containment

Anything that resolves is logged once, at INFO, with both labels, so a run
log always shows which physical object a cached tree actually drove.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from spark_real.bt_library import _canon_word, _norm

logger = logging.getLogger(__name__)

_INSTANCE_SUFFIX = re.compile(r"\s+(\d+)\s*$")


def base_label(label: Any) -> str:
    """
    Strip a trailing instance number: ``"fork 2"`` -> ``"fork"``.
    """
    return _INSTANCE_SUFFIX.sub("", str(label)).strip()


def instance_index(label: Any) -> Optional[int]:
    """
    The trailing instance number, or None for a bare label.
    """
    m = _INSTANCE_SUFFIX.search(str(label))
    return int(m.group(1)) if m else None


def canon_label(label: Any) -> str:
    """
    Spelling- and plural-folded form of a label, instance suffix dropped.

    Uses the same word canonicaliser as the BT cache key, so ``grey block``,
    ``gray blocks`` and ``gray block 2`` all fold to ``gray block``.
    """
    return " ".join(_canon_word(w) for w in _norm(base_label(label)).split() if w)


class LabelResolvingDetectionMap(dict):
    """
    Detection map that tolerates label drift between a cached BT and the
    scene it is replayed in.

    Drop-in for the plain dict the executor already receives: ``get``,
    ``__getitem__`` and ``__contains__`` resolve through the fallback chain,
    while iteration, ``keys()`` and ``items()`` continue to expose exactly
    the real detected labels (so anything that enumerates the map -- the
    destination filter, the obstacle push -- sees the true scene and not a
    set of synonyms).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # label asked for -> label actually served. Reported in the run
        # metadata so a mis-binding is visible after the fact.
        self.resolutions: Dict[str, str] = {}
        # Declared role alias -> real detected label. Populated from the
        # perception layer's ObjectDetection.role_labels. Kept OUT of the
        # dict proper so iteration still yields only real detections.
        self.aliases: Dict[str, str] = {}

    def register_alias(self, alias: Any, real_label: Any) -> None:
        """
        Bind a declared role name to a real detected label.

        Roles come from the task prompt registry: for `stack the blocks of
        same color` the staged pair's colour changes between episodes, so one
        cached BT can only address the two cubes through colour-free names
        (`same color block 1` / `... 2`). First registration wins, so an
        alias is stable within a run even if two detections claim it.
        """
        key = str(alias).strip()
        real = str(real_label)
        if not key or key == real:
            return
        self.aliases.setdefault(key, real)

    def register_aliases(self, aliases: Any, real_label: Any) -> None:
        """Register every entry of a detection's ``role_labels`` list."""
        for alias in aliases or ():
            self.register_alias(alias, real_label)

    # resolution

    def _by_canon(self) -> Dict[str, List[Tuple[Optional[int], str]]]:
        """
        canonical base label -> [(instance index or None, real label)],
        ordered by instance index with bare labels first.
        """
        buckets: Dict[str, List[Tuple[Optional[int], str]]] = {}
        for real in super().keys():
            buckets.setdefault(canon_label(real), []).append((instance_index(real), str(real)))
        for entries in buckets.values():
            entries.sort(key=lambda t: (t[0] is not None, t[0] if t[0] else 0))
        return buckets

    def resolve_label(self, label: Any) -> Optional[str]:
        """
        The real detected label this BT label should bind to, or None.
        """
        key = str(label)
        if dict.__contains__(self, key):
            return key
        if not key.strip():
            return None

        # Declared role aliases outrank every heuristic below: they were
        # published by the task's perception contract, not inferred.
        real = self.aliases.get(key)
        if real is not None and dict.__contains__(self, real):
            return real

        buckets = self._by_canon()
        want_canon = canon_label(key)
        want_idx = instance_index(key)

        # 1. Same canonical base. Covers grey/gray, plurals, and the
        #    arity break in both directions.
        entries = buckets.get(want_canon)
        if entries:
            if want_idx is not None:
                for idx, real in entries:
                    if idx == want_idx:
                        return real
                # "blue block 2" asked for, only one blue block present:
                # bind to the sole instance rather than hard-failing.
                if len(entries) == 1:
                    return entries[0][1]
                # Several present but not that index: take the lowest, so
                # the choice is deterministic rather than dict-order.
                return entries[0][1]
            # Bare label asked for, numbered instances present: bind to the
            # lowest-numbered one. Deterministic given the prompt registry's
            # geometric ordering, so run N and run N+1 pick the same object.
            return entries[0][1]

        # 2. Sub-part / super-part containment, only when unambiguous.
        #    "knife" -> "knife handle", or "knife handle" -> "knife".
        want_words = want_canon.split()
        if want_words:
            hits = [
                real
                for canon, ents in buckets.items()
                for _, real in ents
                if canon != want_canon
                and (
                    _is_subsequence(want_words, canon.split())
                    or _is_subsequence(canon.split(), want_words)
                )
            ]
            if len(set(canon_label(h) for h in hits)) == 1:
                return sorted(hits)[0]

        return None

    def _served(self, label: Any) -> Optional[str]:
        real = self.resolve_label(label)
        key = str(label)
        if real is not None and real != key and key not in self.resolutions:
            self.resolutions[key] = real
            logger.info(
                "label resolver: cached BT label %r -> detection %r "
                "(exact label not in this scene)",
                key,
                real,
            )
        return real

    # dict surface used by the executor / skills

    def get(self, label, default=None):
        real = self._served(label)
        return dict.get(self, real, default) if real is not None else default

    def __getitem__(self, label):
        real = self._served(label)
        if real is None:
            raise KeyError(label)
        return dict.__getitem__(self, real)

    def __contains__(self, label) -> bool:
        return self._served(label) is not None


def _is_subsequence(needle: List[str], haystack: List[str]) -> bool:
    """
    True when every word of ``needle`` appears in ``haystack`` in order.
    ``["knife"]`` is a subsequence of ``["knife", "handle"]``.
    """
    if not needle or len(needle) > len(haystack):
        return False
    it = iter(haystack)
    return all(word in it for word in needle)


__all__ = [
    "LabelResolvingDetectionMap",
    "base_label",
    "canon_label",
    "instance_index",
]
