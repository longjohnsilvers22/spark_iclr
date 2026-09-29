"""
Voyager-style BT library for spark_real.

Persists behaviour-tree scores from real-robot trials to disk and serves them
back, either as few-shot examples for the planner or as the executed plan
itself when the LLM is disabled.

Layout::

    bt_library/
        index.json           -> [{hash, instruction, success, fail}, ...]
        runtime_state.json   -> {use_cached_bt, pins}
        <hash>.json          -> {instruction, score, objects, success, fail, ...}

The hash is content-addressed (instruction + canonicalised score) so the
same plan executed twice merges into one entry with bumped success/fail
counters.

Retrieval is *not* fuzzy-first. ``lookup()`` resolves in this order:

    1. per-task pin  -- an operator/registry binding of one instruction to
       one entry hash; persisted, and NOT consumed on read, so N consecutive
       runs of the same task serve the same tree.
    2. exact match   -- normalised instruction equality against an entry's
       ``instruction`` or any of its ``aliases``.
    3. similarity    -- Jaccard over content words, gated by
       ``min_similarity`` (default 0.60). Below the floor is a MISS, not a
       hit (at ``sim > 0``, "put the pen in the bin" matched a tray plan at
       0.20).

Usage::

    lib = BTLibrary(Path("~/spark/bt_library").expanduser())
    match = lib.lookup("put the knife in the tray")
    if match is not None:
        score, source = match.score, match.source   # source: pin|exact|alias|similar
    examples = lib.top_k_for(instruction, k=3)      # -> List[(instruction, score)]
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Similarity floor for the fuzzy fallback. Overridable per-library via the
# constructor (INTEGRATOR feeds ``bt.min_similarity`` from the family yaml).
#
# Deliberately high. Instruction variance is meant to be absorbed by the
# exact and near-exact tiers (aliases + spelling/plural canonicalisation),
# NOT by Jaccard: at a low floor "put the pen in the bin" scores 0.20
# against a tray plan and would execute it.
DEFAULT_MIN_SIMILARITY = 0.75

# Successes (with zero failures) after which an entry auto-promotes.
# Overridable per-library; INTEGRATOR feeds ``bt.auto_promote_after``.
DEFAULT_AUTO_PROMOTE_AFTER = 3

# Spelling / regional variants folded together before near-exact matching.
# Applied token-wise, so "grey block" and "gray block" are one cache key.
# Keep this list short and unambiguous -- it is not a synonym engine.
_SPELLING_VARIANTS = {
    "grey": "gray",
    "colour": "color",
    "coloured": "colored",
    "cutlery": "silverware",
    "utensil": "silverware",
    "cube": "block",
}

# Filler that never changes which objects a task touches. Dropped by the
# near-exact key so "pick up the knife and place it in the tray" and
# "put the knife in the tray" collapse to the same key.
_NEAR_EXACT_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "in",
        "into",
        "onto",
        "on",
        "at",
        "of",
        "for",
        "and",
        "or",
        "with",
        "by",
        "is",
        "be",
        "it",
        "them",
        "then",
        "please",
        "all",
        "up",
        "down",
        "put",
        "place",
        "pick",
        "set",
        "move",
        "other",
        "another",
        "second",
        "next",
        "its",
        "their",
    }
)

# Prepositions whose ORDER carries the task semantics. Any instruction
# containing one of these is refused a Jaccard match unless the ordered
# content tokens also agree -- "stack the blue block on the gray block"
# and "stack the gray block on the blue block" have identical token SETS
# (jaccard 1.0) and opposite meanings.
_POSITIONAL_PREPOSITIONS = frozenset(
    {"on", "in", "into", "onto", "under", "above", "below", "beside", "behind"}
)

# Score keys that carry per-run bookkeeping rather than plan semantics. They
# are stripped before hashing so a plan served from cache re-hashes to the
# entry it came from:
#   - label_corrections: popped by pipeline_execution.execute() before the
#     score is handed to the executor, so the score written back after a run
#     differs from the one served.
#   - __-prefixed keys (__raw_yaml, __planner): planner provenance; the raw
#     YAML text differs in whitespace for byte-identical trees.
#   - verify: the success PREDICATE, not the plan. Two runs that drive the
#     same motion are the same tree whether or not the planner also said how
#     to check the result, so a predicate must never split a cache entry.
_NONSEMANTIC_KEYS = ("label_corrections", "verify")

# Params whose documented default is a no-op. A plan that spells out the
# default must hash identically to the cached tree that omits it, or every
# cached entry misses when the planner emits the new schema.
# Reference semantics: planning/spark_planner.normalize_plan_for_hash.
_DEFAULT_PARAMS = {"grasp_strategy": "auto", "grasp_yaw_deg": None}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "from",
        "in",
        "on",
        "at",
        "of",
        "for",
        "and",
        "or",
        "with",
        "by",
        "is",
        "be",
        "this",
        "that",
        "it",
        "into",
        "onto",
    }
)


def _norm(text: Any) -> str:
    """
    Canonical form of an instruction string for equality comparison.

    Lowercase, underscores and punctuation to spaces, whitespace collapsed.
    The single normalisation dialect in the codebase, exported under the
    public name :func:`normalize_instruction` (bottom of this module); every
    other module calls that one function. ``_norm`` is the in-module name.
    """
    cleaned = re.sub(r"[^\w\s]", " ", str(text).replace("_", " ").lower())
    return " ".join(cleaned.split())


def _tokens(text: str) -> set:
    """
    Lowercase, strip punctuation, drop stopwords.
    """
    return {w for w in _norm(text).split() if w and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _singular(word: str) -> str:
    """
    Crude, deliberately conservative de-pluralisation.

    Only strips a trailing "s" from words longer than three characters that
    do not end in a doubled or ambiguous sibilant, so "blocks"->"block" and
    "spoons"->"spoon" while "glass", "status" and "this" are left alone.
    Good enough to make "put the spoons in the tray" and "put the spoon in
    the tray" one cache key; it is not a stemmer and must not become one.
    """
    w = str(word)
    if len(w) <= 3 or not w.endswith("s"):
        return w
    if w.endswith(("ss", "us", "is", "es")):
        return w[:-2] if w.endswith("es") and len(w) > 4 else w
    return w[:-1]


def _canon_word(word: str) -> str:
    """
    Fold one token to its canonical spelling: regional variants first, then
    de-pluralisation, then variants again (so "cubes"->"cube"->"block").
    """
    w = _SPELLING_VARIANTS.get(word, word)
    w = _singular(w)
    return _SPELLING_VARIANTS.get(w, w)


def _near_exact_key(text: str) -> str:
    """
    Order-PRESERVING canonical key for the near-exact retrieval tier.

    Drops filler that never changes which objects a task touches, folds
    spelling variants and plurals, and joins what is left *in order*.
    Order matters: "stack the blue block on the gray block" and "stack the
    gray block on the blue block" have identical token sets but opposite
    meanings, so a set-based key would merge two distinct tasks.

        "put the knife in the tray"                -> "knife tray"
        "pick up the knife and place it in the tray" -> "knife tray"
        "put the spoons in the tray"               -> "spoon tray"
        "stack the blue block on the gray block"   -> "stack blue block gray block"
        "stack the gray block on the blue block"   -> "stack gray block blue block"
    """
    words = [_canon_word(w) for w in _norm(text).split() if w]
    kept = [w for w in words if w and w not in _NEAR_EXACT_FILLER]
    return " ".join(kept)


def _ordered_tokens(text: str) -> List[str]:
    """
    Content tokens in source order, canonicalised. Used to reject a Jaccard
    match whose token multiset agrees but whose ORDER does not.
    """
    return [_canon_word(w) for w in _norm(text).split() if w and w not in _STOPWORDS]


def _is_order_sensitive(text: str) -> bool:
    """
    True when the instruction contains a preposition whose position carries
    the task semantics (A on B vs B on A).
    """
    return bool(set(_norm(text).split()) & _POSITIONAL_PREPOSITIONS)


def _strip_nonsemantic(score: Any) -> Any:
    """
    Deep-copy ``score`` with per-run bookkeeping keys removed.

    Applied at every level, not just the root, so a nested correction map
    cannot shift the hash either.
    """
    if isinstance(score, dict):
        return {
            k: _strip_nonsemantic(v)
            for k, v in score.items()
            if k not in _NONSEMANTIC_KEYS
            and not str(k).startswith("__")
            and not (k in _DEFAULT_PARAMS and v == _DEFAULT_PARAMS[k])
        }
    if isinstance(score, list):
        return [_strip_nonsemantic(v) for v in score]
    return score


def _canonical_score_str(score: Any) -> str:
    """
    Stable string repr of a score for hashing, key order insensitive and
    independent of per-run bookkeeping keys.
    """
    return json.dumps(_strip_nonsemantic(score), sort_keys=True, separators=(",", ":"))


def _hash_entry(instruction: str, score: Any) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(_norm(instruction).encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical_score_str(score).encode("utf-8"))
    return h.hexdigest()


@dataclass
class BTEntry:
    hash: str
    instruction: str
    score: Any
    objects: List[str] = field(default_factory=list)
    success: int = 0
    fail: int = 0
    # Runs that executed this tree but produced no verdict (all cameras
    # abstained, occluded, no predicate derivable). Deliberately NOT folded
    # into fail: a systematically-abstaining verifier must make the cache go
    # cold and visible, not wrong. Surfaced on /scores.
    unverified: int = 0
    promoted: bool = False  # Toggled via /api/bt/promote.
    # Extra instruction spellings that resolve to this entry by exact match.
    aliases: List[str] = field(default_factory=list)
    # True for entries installed from configs/bt_seeds/*.yaml. Seeds are
    # authored and version-controlled; run-accumulated entries are not.
    seed: bool = False
    # Author-controlled ordering among entries that match equally well.
    # Higher wins. Run-accumulated entries keep 0.
    priority: int = 0
    # Free-text provenance from the seed file, surfaced in the UI.
    notes: str = ""
    # False when a seed's score has never been confirmed on the real rig.
    verified: bool = True

    def keys(self) -> List[str]:
        """
        Every normalised instruction string that resolves to this entry.
        """
        out = [_norm(self.instruction)]
        out.extend(_norm(a) for a in self.aliases or [])
        return [k for k in out if k]

    def near_keys(self) -> List[str]:
        """
        Every near-exact key (spelling/plural/filler folded, order kept)
        that resolves to this entry.
        """
        out = [_near_exact_key(self.instruction)]
        out.extend(_near_exact_key(a) for a in self.aliases or [])
        return [k for k in out if k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "instruction": self.instruction,
            "score": self.score,
            "objects": self.objects,
            "success": self.success,
            "fail": self.fail,
            "unverified": self.unverified,
            "promoted": self.promoted,
            "aliases": self.aliases,
            "seed": self.seed,
            "priority": self.priority,
            "notes": self.notes,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BTEntry":
        return cls(
            hash=d["hash"],
            instruction=d["instruction"],
            score=d["score"],
            objects=list(d.get("objects") or []),
            success=int(d.get("success", 0)),
            fail=int(d.get("fail", 0)),
            unverified=int(d.get("unverified", 0)),
            promoted=bool(d.get("promoted", False)),
            aliases=list(d.get("aliases") or []),
            seed=bool(d.get("seed", False)),
            priority=int(d.get("priority", 0)),
            notes=str(d.get("notes") or ""),
            verified=bool(d.get("verified", True)),
        )


@dataclass
class BTMatch:
    """
    Result of a successful ``BTLibrary.lookup``.
    """

    entry: BTEntry
    source: str  # pin | exact | alias | normalized | similar | seed
    similarity: float

    @property
    def score(self) -> Any:
        """
        Detached copy, so a downstream ``pop()`` cannot mutate the library.
        """
        return copy.deepcopy(self.entry.score)

    @property
    def hash(self) -> str:
        return self.entry.hash


class BTCacheMiss(LookupError):
    """
    Raised by ``resolve`` when no cached BT covers the instruction and
    strict offline mode (``bt.require_cached`` / SPARK_REQUIRE_CACHED_BT)
    forbids falling through to the LLM.
    """


class BTLibrary:
    """
    Disk-backed library of BT scores keyed by instruction.
    """

    def __init__(
        self,
        root: Path,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        auto_promote_after: int = DEFAULT_AUTO_PROMOTE_AFTER,
    ):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.state_path = self.root / "runtime_state.json"
        self.min_similarity = float(min_similarity)
        self.auto_promote_after = int(auto_promote_after)
        self._entries: Dict[str, BTEntry] = {}
        # Persisted runtime state (survives server restarts).
        self._pins: Dict[str, str] = {}
        self._use_cached_bt: Optional[bool] = None
        self._load()
        self._load_runtime_state()

    # I/O

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            blob = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("BTLibrary: bad index %s: %s; starting fresh", self.index_path, e)
            return
        for stub in blob:
            h = stub.get("hash")
            if not h:
                continue
            entry_path = self.root / f"{h}.json"
            if not entry_path.exists():
                continue
            try:
                self._entries[h] = BTEntry.from_dict(json.loads(entry_path.read_text()))
            except Exception as e:
                logger.warning("BTLibrary: skipping %s: %s", entry_path, e)

    def _write_entry(self, entry: BTEntry) -> None:
        (self.root / f"{entry.hash}.json").write_text(json.dumps(entry.to_dict(), indent=2))

    def _flush(self) -> None:
        index = [
            {
                "hash": e.hash,
                "instruction": e.instruction,
                "success": e.success,
                "fail": e.fail,
                "promoted": e.promoted,
                "seed": e.seed,
            }
            for e in self._entries.values()
        ]
        self.index_path.write_text(json.dumps(index, indent=2))

    # persisted runtime state

    def _load_runtime_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            blob = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("BTLibrary: bad runtime state %s: %s", self.state_path, e)
            return
        raw_pins = blob.get("pins") or {}
        self._pins = {_norm(k): str(v) for k, v in raw_pins.items() if _norm(k) and v}
        cached = blob.get("use_cached_bt")
        self._use_cached_bt = None if cached is None else bool(cached)

    def _flush_runtime_state(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {"use_cached_bt": self._use_cached_bt, "pins": self._pins},
                indent=2,
            )
        )

    @property
    def use_cached_bt(self) -> Optional[bool]:
        """
        Persisted cache-only override, or None to defer to env/config.
        """
        return self._use_cached_bt

    def set_use_cached_bt(self, value: Optional[bool]) -> Optional[bool]:
        self._use_cached_bt = None if value is None else bool(value)
        self._flush_runtime_state()
        return self._use_cached_bt

    # per-task pins

    def pins(self) -> Dict[str, str]:
        """
        Normalised instruction -> entry hash. Copy; mutate via pin/unpin.
        """
        return dict(self._pins)

    def pin(self, instruction: str, h: str) -> Optional[BTEntry]:
        """
        Bind one instruction to one entry. Persisted, and not consumed on
        read, so 30 consecutive runs of a task serve the same tree.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        key = _norm(instruction)
        if not key:
            return None
        self._pins[key] = h
        self._flush_runtime_state()
        return entry

    def unpin(self, instruction: str) -> Optional[str]:
        prev = self._pins.pop(_norm(instruction), None)
        if prev is not None:
            self._flush_runtime_state()
        return prev

    def pinned_for(self, instruction: str) -> Optional[BTEntry]:
        h = self._pins.get(_norm(instruction))
        if h is None:
            return None
        entry = self._entries.get(h)
        if entry is None:
            # Pin points at a deleted entry: drop it rather than silently
            # falling through on every future call.
            self._pins.pop(_norm(instruction), None)
            self._flush_runtime_state()
            logger.warning(
                "BTLibrary: pin for %r referenced missing entry %s; cleared",
                instruction,
                h,
            )
        return entry

    # promotion

    def set_promoted(self, h: str, promoted: bool) -> Optional[BTEntry]:
        """
        Toggle the ``promoted`` flag on an entry by hash.

        Promoted entries are always injected into the planner prompt
        (independent of jaccard similarity to the current instruction),
        so they act as canonical recipes the LLM can rely on. Regular
        entries only show up when the current instruction is similar
        to the entry's instruction.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        entry.promoted = bool(promoted)
        self._write_entry(entry)
        self._flush()
        return entry

    def promoted_entries(self) -> List[Tuple[str, Any]]:
        """
        Return (instruction, score) for every promoted entry. Used
        by the planner to always-include these as few-shot examples
        in addition to similarity-retrieved ones.
        """
        return [
            (e.instruction, e.score)
            for e in self._entries.values()
            if e.promoted and e.success >= 1
        ]

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Index view for the frontend library panel.
        """
        pinned = set(self._pins.values())
        out = []
        for e in self._entries.values():
            out.append(
                {
                    "hash": e.hash,
                    "instruction": e.instruction,
                    "aliases": e.aliases,
                    "objects": e.objects,
                    "success": e.success,
                    "fail": e.fail,
                    "promoted": e.promoted,
                    "seed": e.seed,
                    "priority": e.priority,
                    "verified": e.verified,
                    "notes": e.notes,
                    "pinned": e.hash in pinned,
                }
            )
        out.sort(key=lambda x: (not x["seed"], not x["promoted"], -x["success"]))
        return out

    def get(self, h: str) -> Optional[BTEntry]:
        return self._entries.get(h)

    def get_score(self, h: str) -> Optional[Any]:
        e = self._entries.get(h)
        return e.score if e is not None else None

    def entries(self) -> Iterable[BTEntry]:
        return self._entries.values()

    # mutation

    def add(
        self,
        instruction: str,
        score: Any,
        objects: Optional[List[str]] = None,
        success: bool = True,
        aliases: Optional[List[str]] = None,
        seed: bool = False,
        priority: int = 0,
        notes: str = "",
        verified: bool = True,
    ) -> BTEntry:
        """
        Insert or update an entry; bumps the success/fail counter.

        Callers that *served* a plan from this library must use ``bump()``
        instead: re-adding a cache hit under a slightly different
        instruction string mints a near-duplicate entry.
        """
        h = _hash_entry(instruction, score)
        if h in self._entries:
            entry = self._entries[h]
            if success:
                entry.success += 1
            else:
                entry.fail += 1
            # Seed metadata is authoritative when re-seeding an existing row.
            if seed:
                entry.seed = True
                entry.priority = priority
                entry.notes = notes or entry.notes
                entry.verified = verified
                for alias in aliases or []:
                    if alias not in entry.aliases:
                        entry.aliases.append(alias)
        else:
            entry = BTEntry(
                hash=h,
                instruction=instruction,
                score=score,
                objects=list(objects or []),
                success=1 if success else 0,
                fail=0 if success else 1,
                aliases=list(aliases or []),
                seed=bool(seed),
                priority=int(priority),
                notes=str(notes or ""),
                verified=bool(verified),
            )
            self._entries[h] = entry
        if not seed:
            self._maybe_auto_promote(entry)
        self._write_entry(entry)
        self._flush()
        return entry

    def bump(
        self,
        h: str,
        success: bool,
        alias: Optional[str] = None,
    ) -> Optional[BTEntry]:
        """
        Record an outcome against an entry that already exists, without
        minting a new row. This is the write path for every run whose plan
        was SERVED from the library (pin / exact / normalized / similar).

        ``add()`` would instead mint a near-duplicate keyed on the run's
        phrasing (the live library once grew to 88 rows for ~20 distinct
        tasks). Passing the run's phrasing as ``alias`` records it on the
        entry that ran, so the next run of that phrasing is an exact hit.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        if success:
            entry.success += 1
        else:
            entry.fail += 1
        if alias:
            self._note_alias(entry, alias)
        self._maybe_auto_promote(entry)
        self._write_entry(entry)
        self._flush()
        return entry

    def note_unverified(
        self,
        h: Optional[str],
        instruction: Optional[str] = None,
        score: Any = None,
        objects: Optional[List[str]] = None,
    ) -> Optional[BTEntry]:
        """
        Record that a run of this entry produced NO verdict.

        Bumps neither success nor fail: an unverified run must not reinforce a
        tree and must not penalise one either. It makes the verifier's blind
        spots countable on /scores.

        Mints the row when the hash is unknown and the run's material is
        supplied, so an LLM-planned run (no ``bt_hash`` yet) that the verifier
        never resolves still accumulates a counter. A minted row carries
        success=0, below ``lookup``'s ``min_success=1`` floor, so it is
        countable without becoming cache-eligible.
        """
        entry = self._entries.get(h) if h else None
        if entry is None:
            if not instruction or not score:
                return None
            h2 = _hash_entry(instruction, score)
            entry = self._entries.get(h2)
            if entry is None:
                entry = BTEntry(
                    hash=h2,
                    instruction=instruction,
                    score=score,
                    objects=list(objects or []),
                )
                self._entries[h2] = entry
        entry.unverified += 1
        self._write_entry(entry)
        self._flush()
        return entry

    def _note_alias(self, entry: BTEntry, phrasing: str) -> bool:
        """
        Record an extra instruction spelling on an entry, if it is not
        already one of its keys. Returns True when something was added.
        """
        key = _norm(phrasing)
        if not key or key in entry.keys():
            return False
        entry.aliases.append(str(phrasing))
        logger.info(
            "BTLibrary: %s learned alias %r (now %d spelling(s))",
            entry.hash,
            phrasing,
            len(entry.keys()),
        )
        return True

    def alias(self, h: str, phrasing: str) -> Optional[BTEntry]:
        """
        Bind an extra instruction spelling to an existing entry.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        if self._note_alias(entry, phrasing):
            self._write_entry(entry)
            self._flush()
        return entry

    def _maybe_auto_promote(self, entry: BTEntry) -> bool:
        """
        Promote an entry that has proven itself. Caller flushes.
        """
        if entry.promoted or self.auto_promote_after <= 0:
            return False
        if entry.success >= self.auto_promote_after and entry.fail == 0:
            entry.promoted = True
            logger.info(
                "BTLibrary: auto-promoted %s (%r) at %d/%d successes",
                entry.hash,
                entry.instruction,
                entry.success,
                self.auto_promote_after,
            )
            return True
        return False

    def delete_by_instruction(self, instruction: str) -> List[str]:
        """
        Delete every entry keyed on ``instruction`` (exact or alias).
        Returns the hashes removed. Pruning tool for the operator: a task
        whose cached tree turned out to be wrong is one call to clear.
        """
        doomed = [e.hash for e in self.exact_matches(instruction, min_success=0)]
        for h in doomed:
            self.delete(h)
        return doomed

    def delete(self, h: str) -> Optional[BTEntry]:
        """
        Remove an entry from disk and the in-memory index, and drop any
        pins that referenced it.
        """
        entry = self._entries.pop(h, None)
        if entry is None:
            return None
        path = self.root / f"{h}.json"
        if path.exists():
            path.unlink()
        self._flush()
        stale = [k for k, v in self._pins.items() if v == h]
        if stale:
            for k in stale:
                self._pins.pop(k, None)
            self._flush_runtime_state()
        return entry

    # retrieval

    def _rank_key(self, entry: BTEntry) -> Tuple:
        """
        Total, stable order among equally-matching entries.

        Trailing hash makes it a total order; without it, ties fall back to
        index.json insertion order and the tree served for a task can flip
        mid-session.
        """
        return (
            entry.priority,
            int(entry.promoted),
            entry.success,
            -entry.fail,
            entry.hash,
        )

    def exact_matches(self, instruction: str, min_success: int = 0) -> List[BTEntry]:
        """
        Every entry whose instruction or alias normalises to ``instruction``,
        best first.
        """
        key = _norm(instruction)
        if not key:
            return []
        hits = [e for e in self._entries.values() if key in e.keys() and e.success >= min_success]
        hits.sort(key=self._rank_key, reverse=True)
        return hits

    def near_exact_matches(self, instruction: str, min_success: int = 0) -> List[BTEntry]:
        """
        Entries matching on the near-exact key: spelling variants, plurals
        and phrasing filler folded away, token ORDER preserved.

        This is the tier that absorbs everyday instruction variance --
        "put the spoons in the tray" / "pick up the spoon and place it in
        the tray" / "put the spoon in the tray" are one task -- without
        letting Jaccard merge genuinely different ones.
        """
        key = _near_exact_key(instruction)
        if not key:
            return []
        hits = [
            e for e in self._entries.values() if key in e.near_keys() and e.success >= min_success
        ]
        hits.sort(key=self._rank_key, reverse=True)
        return hits

    def lookup(
        self,
        instruction: str,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> Optional[BTMatch]:
        """
        Resolve an instruction to a single BT. Four tiers, best first:

            1. ``pin``        -- persisted per-task binding, never consumed.
            2. ``exact``/``alias`` -- normalised string equality.
            3. ``normalized`` -- near-exact key (spelling, plurals, filler).
            4. ``similar``    -- Jaccard >= floor, gated on token ORDER when
               the instruction contains a positional preposition.

        None on a miss. Below the similarity floor is a MISS, not a hit.
        """
        entry = self.pinned_for(instruction)
        if entry is not None:
            return BTMatch(entry=entry, source="pin", similarity=1.0)

        key = _norm(instruction)
        for hit in self.exact_matches(instruction, min_success=min_success):
            source = "exact" if _norm(hit.instruction) == key else "alias"
            return BTMatch(entry=hit, source=source, similarity=1.0)

        for hit in self.near_exact_matches(instruction, min_success=min_success):
            return BTMatch(entry=hit, source="normalized", similarity=1.0)

        floor = self.min_similarity if min_similarity is None else float(min_similarity)
        order_sensitive = _is_order_sensitive(instruction)
        query_ordered = _ordered_tokens(instruction)
        query = set(query_ordered)
        best: Optional[Tuple[float, Tuple, BTEntry]] = None
        for candidate in self._entries.values():
            if candidate.success < min_success:
                continue
            sim = _jaccard(query, set(_ordered_tokens(candidate.instruction)))
            if sim < floor:
                continue
            # Order gate: a set-similarity hit on an order-sensitive
            # instruction (see the positional-preposition list above) is only
            # trustworthy when the ordered token lists agree as well.
            if order_sensitive or _is_order_sensitive(candidate.instruction):
                if query_ordered != _ordered_tokens(candidate.instruction):
                    logger.debug(
                        "BTLibrary: rejecting order-sensitive fuzzy match "
                        "%r ~ %r (sim=%.2f, token order differs)",
                        instruction,
                        candidate.instruction,
                        sim,
                    )
                    continue
            ranked = (sim, self._rank_key(candidate), candidate)
            if best is None or (ranked[0], ranked[1]) > (best[0], best[1]):
                best = ranked
        if best is None:
            return None
        return BTMatch(entry=best[2], source="similar", similarity=best[0])

    def resolve(
        self,
        instruction: str,
        require_cached: bool = False,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> Optional[BTMatch]:
        """
        ``lookup`` plus strict offline mode.

        With ``require_cached`` a miss raises ``BTCacheMiss`` instead of
        letting the caller fall through to the LLM. Demo collection runs
        with this on: a silent Gemini call mid-collection produces an
        episode whose plan nobody froze.
        """
        match = self.lookup(
            instruction,
            min_success=min_success,
            min_similarity=min_similarity,
        )
        if match is None and require_cached:
            raise BTCacheMiss(
                f"no cached BT for {instruction!r} "
                f"(floor={self.min_similarity:.2f}, entries={len(self._entries)}); "
                "seed one into the library or clear bt.require_cached"
            )
        return match

    def top_k_for(
        self,
        instruction: str,
        k: int = 3,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, Any]]:
        """
        Up to k (instruction, score) pairs for few-shot prompting, best
        first. Exact/alias matches always outrank similarity matches.

        ``min_similarity=0.0`` disables the floor (any shared content word
        matches); the default applies it.
        """
        floor = self.min_similarity if min_similarity is None else float(min_similarity)
        picked: List[BTEntry] = []
        seen: set = set()
        for hit in self.exact_matches(instruction, min_success=min_success):
            picked.append(hit)
            seen.add(hit.hash)
        for hit in self.near_exact_matches(instruction, min_success=min_success):
            if hit.hash not in seen:
                picked.append(hit)
                seen.add(hit.hash)

        query = set(_ordered_tokens(instruction))
        scored: List[Tuple[float, Tuple, BTEntry]] = []
        for entry in self._entries.values():
            if entry.hash in seen or entry.success < min_success:
                continue
            sim = _jaccard(query, set(_ordered_tokens(entry.instruction)))
            if sim < floor or sim <= 0:
                continue
            scored.append((sim, self._rank_key(entry), entry))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        picked.extend(e for _, _, e in scored)
        return [(e.instruction, e.score) for e in picked[:k]]

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> Dict[str, int]:
        total_success = sum(e.success for e in self._entries.values())
        total_fail = sum(e.fail for e in self._entries.values())
        return {
            "entries": len(self._entries),
            "seeds": sum(1 for e in self._entries.values() if e.seed),
            "pins": len(self._pins),
            "total_success": total_success,
            "total_fail": total_fail,
        }


# --------------------------------------------------------------------------
# Public names for the instruction-matching dialect. Every stage that keys off
# the instruction string must agree with the BT cache on "the same
# instruction", or a task can hit its cached tree and miss its SAM3 prompts.
# The underscore-prefixed originals are in-module short names, not the API.
#
#   normalize_instruction -- tier 2 key (exact equality after lowering,
#                            underscores/punctuation to spaces, whitespace
#                            collapsed).
#   near_exact_key        -- tier 3 key (spelling variants, plurals and filler
#                            folded, content-word ORDER preserved, so
#                            "A on B" and "B on A" stay distinct).
# --------------------------------------------------------------------------
normalize_instruction = _norm
near_exact_key = _near_exact_key

__all__ = [
    "BTLibrary",
    "BTEntry",
    "BTMatch",
    "BTCacheMiss",
    "DEFAULT_MIN_SIMILARITY",
    "DEFAULT_AUTO_PROMOTE_AFTER",
    "normalize_instruction",
    "near_exact_key",
]
