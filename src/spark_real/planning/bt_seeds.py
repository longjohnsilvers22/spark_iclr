"""
Version-controlled BT seeds -> BTLibrary.

Seeds are one YAML file per task under ``configs/bt_seeds/``, shipped with
the repo and loaded into the library at boot (idempotent: ``add()`` de-dups
on the content hash), so a fresh checkout serves its tasks without a Gemini
call.

File schema (all keys optional except ``instruction`` and ``score``)::

    instruction: "put the knife in the tray"     # authoritative cache key
    aliases: ["put knife in tray"]               # extra exact-match keys
    objects: ["knife handle", "tray"]            # labels the score expects
    score:                                       # the BT itself
      task: "put the knife in the tray"
      tree: {type: sequence, children: [...]}
    priority: 10        # higher wins among entries matching equally well
    promote: true       # also inject as a planner few-shot example
    verified: true      # false => score never confirmed on the real rig
    provenance: "..."   # where the tree came from
    notes: "..."        # surfaced in the library UI

A file carrying ``todo: true`` is parsed, reported, and deliberately NOT
installed -- that is the marker for a task whose score could not be
constructed with confidence.

Entry points::

    load_seeds(seed_dir=None) -> List[Seed]
    seed_library(lib, seeds=None, seed_dir=None, force=False) -> SeedReport
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from spark_real.bt_library import BTEntry, BTLibrary, _hash_entry, _norm

logger = logging.getLogger(__name__)

# configs/ lives next to this package's config loader; keep one definition.
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Relative to CONFIGS_DIR. Overridable via config (bt.seed_dir) or the
# SPARK_BT_SEEDS env var, which may also be an absolute path.
DEFAULT_SEED_SUBDIR = "bt_seeds"
SEED_DIR_ENV = "SPARK_BT_SEEDS"

# Control-flow node types; everything else is a leaf skill invocation.
_CONTROL_TYPES = frozenset({"sequence", "selector", "fallback", "parallel"})


class SeedError(ValueError):
    """
    A seed file exists but is not loadable as a BT seed.
    """


@dataclass
class Seed:
    instruction: str
    score: Dict[str, Any]
    path: Optional[Path] = None
    aliases: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    priority: int = 0
    promote: bool = True
    verified: bool = True
    provenance: str = ""
    notes: str = ""
    todo: bool = False
    todo_reason: str = ""

    @property
    def slug(self) -> str:
        return self.path.stem if self.path is not None else _norm(self.instruction)

    def action_types(self) -> List[str]:
        """
        Leaf node types the score dispatches, in tree order. Used to check
        a seed against the live skill registry without executing anything.
        """
        out: List[str] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            kids = node.get("children")
            if isinstance(kids, list) and kids:
                for child in kids:
                    walk(child)
                return
            node_type = node.get("type")
            if node_type and node_type not in _CONTROL_TYPES:
                out.append(str(node_type))

        walk(self.score.get("tree"))
        return out

    def keypoint_labels(self) -> List[str]:
        """
        Every detection label the score references, deduplicated in tree
        order. These must survive re-detection for the seed to replay.
        """
        out: List[str] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            params = node.get("params") or {}
            if isinstance(params, dict):
                for key in ("keypoint_label", "target_label", "container_label"):
                    val = params.get(key)
                    if isinstance(val, str) and val and val not in out:
                        out.append(val)
            for child in node.get("children") or []:
                walk(child)

        walk(self.score.get("tree"))
        # Predicate labels live outside the tree but must survive re-detection
        # too: a seed whose verify block names an unprompted label executes
        # fine and then cannot be judged.
        block = self.score.get("verify")
        if isinstance(block, dict):
            for pred in block.get("all") or []:
                if not isinstance(pred, dict):
                    continue
                for key in ("obj", "container", "surface", "base", "target"):
                    val = pred.get(key)
                    if isinstance(val, str) and val and val not in out:
                        out.append(val)
        return out


@dataclass
class SeedReport:
    installed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    todo: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "installed": self.installed,
            "skipped": self.skipped,
            "todo": self.todo,
            "errors": self.errors,
            "n_installed": len(self.installed),
        }


def default_seed_dir(seed_dir: Optional[str] = None) -> Path:
    """
    Resolve the seed directory.

    Precedence: explicit argument > ``SPARK_BT_SEEDS`` env > packaged
    ``configs/bt_seeds``. Relative values resolve against ``configs/``, so
    the config key can stay a short in-package name and never a machine path.
    """
    raw = seed_dir or os.environ.get(SEED_DIR_ENV) or DEFAULT_SEED_SUBDIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = CONFIGS_DIR / path
    return path


def parse_seed(blob: Dict[str, Any], path: Optional[Path] = None) -> Seed:
    """
    Validate one parsed YAML document into a Seed.
    """
    where = str(path) if path is not None else "<inline>"
    if not isinstance(blob, dict):
        raise SeedError(f"{where}: top level must be a mapping")

    instruction = blob.get("instruction") or (blob.get("score") or {}).get("task")
    if not isinstance(instruction, str) or not _norm(instruction):
        raise SeedError(f"{where}: missing or empty 'instruction'")

    todo = bool(blob.get("todo", False))
    score = blob.get("score")
    if not todo:
        if not isinstance(score, dict):
            raise SeedError(f"{where}: 'score' must be a mapping")
        tree = score.get("tree")
        if not isinstance(tree, dict) or not tree.get("type"):
            raise SeedError(f"{where}: score.tree must be a mapping with a 'type'")
    score = score if isinstance(score, dict) else {}

    aliases = blob.get("aliases") or []
    if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
        raise SeedError(f"{where}: 'aliases' must be a list of strings")

    objects = blob.get("objects") or []
    if not isinstance(objects, list) or any(not isinstance(o, str) for o in objects):
        raise SeedError(f"{where}: 'objects' must be a list of strings")

    return Seed(
        instruction=instruction,
        score=score,
        path=path,
        aliases=list(aliases),
        objects=list(objects),
        priority=int(blob.get("priority", 0)),
        promote=bool(blob.get("promote", True)),
        verified=bool(blob.get("verified", True)),
        provenance=str(blob.get("provenance") or ""),
        notes=str(blob.get("notes") or ""),
        todo=todo,
        todo_reason=str(blob.get("todo_reason") or ""),
    )


def load_seeds(seed_dir: Optional[str] = None) -> List[Seed]:
    """
    Parse every ``*.yaml`` / ``*.yml`` under the seed directory, sorted by
    filename so the install order is reproducible. Unparseable files raise;
    a missing directory yields an empty list.
    """
    root = default_seed_dir(seed_dir)
    if not root.is_dir():
        logger.info("bt_seeds: no seed directory at %s", root)
        return []
    paths = sorted(p for p in root.iterdir() if p.suffix in (".yaml", ".yml") and p.is_file())
    seeds: List[Seed] = []
    for path in paths:
        blob = yaml.safe_load(path.read_text())
        seeds.append(parse_seed(blob, path=path))
    return seeds


def seed_library(
    lib: BTLibrary,
    seeds: Optional[List[Seed]] = None,
    seed_dir: Optional[str] = None,
    force: bool = False,
) -> SeedReport:
    """
    Install seeds into ``lib``. Idempotent.

    A seed already present (same instruction + same canonical score) is left
    alone rather than re-``add()``ed, because ``add()`` bumps the success
    counter and would inflate a seed's score on every server boot. ``force``
    re-writes the entry metadata (aliases/priority/notes) without touching
    the counters.

    Seeds are installed with ``success=True`` so they clear the
    ``min_success >= 1`` retrieval filter, and promoted so they are also
    available as planner few-shot examples for unregistered instructions.
    """
    report = SeedReport()
    if lib is None:
        report.errors.append("no BT library")
        return report
    if seeds is None:
        try:
            seeds = load_seeds(seed_dir)
        except (SeedError, OSError, yaml.YAMLError) as exc:
            report.errors.append(str(exc))
            logger.error("bt_seeds: load failed: %s", exc)
            return report

    for seed in seeds:
        if seed.todo:
            report.todo.append(seed.slug)
            logger.warning(
                "bt_seeds: %s is marked TODO (%s); not installed",
                seed.slug,
                seed.todo_reason or "no score authored",
            )
            continue
        try:
            existing = _find_existing(lib, seed)
            if existing is not None and not force:
                report.skipped.append(seed.slug)
                continue
            entry = lib.add(
                seed.instruction,
                seed.score,
                objects=seed.objects,
                success=existing is None,
                aliases=seed.aliases,
                seed=True,
                priority=seed.priority,
                notes=(seed.notes or seed.provenance),
                verified=seed.verified,
            )
            if seed.promote:
                lib.set_promoted(entry.hash, True)
            report.installed.append(seed.slug)
            logger.info(
                "bt_seeds: installed %s -> %s (%r)",
                seed.slug,
                entry.hash,
                seed.instruction,
            )
        except Exception as exc:  # one bad seed must not block the rest
            report.errors.append(f"{seed.slug}: {exc}")
            logger.exception("bt_seeds: failed to install %s", seed.slug)
    return report


def _find_existing(lib: BTLibrary, seed: Seed) -> Optional[BTEntry]:
    """
    The entry this seed would produce, if it is already installed.
    """
    return lib.get(_hash_entry(seed.instruction, seed.score))


def ensure_seeded(
    lib: BTLibrary,
    seed_dir: Optional[str] = None,
    force: bool = False,
) -> SeedReport:
    """
    Install seeds once per library instance.

    Idempotent and cheap to call from a request handler: after the first
    call the library carries a ``_seeded`` marker and subsequent calls are a
    no-op. Exists so a fresh checkout serves the demo tasks from cache on
    the very first request, without waiting for a boot-time hook.
    """
    if lib is None:
        return SeedReport(errors=["no BT library"])
    if getattr(lib, "_seeded", False) and not force:
        return SeedReport()
    report = seed_library(lib, seed_dir=seed_dir, force=force)
    lib._seeded = True
    if report.installed or report.todo or report.errors:
        logger.info(
            "bt_seeds: %d installed, %d already present, %d TODO, %d error(s)",
            len(report.installed),
            len(report.skipped),
            len(report.todo),
            len(report.errors),
        )
    return report


def entry_to_seed(entry: BTEntry) -> Dict[str, Any]:
    """
    Render a live library entry as a seed document.

    The round trip that makes "capture once" durable: a tree that proved
    itself during a run gets written out as a version-controlled YAML file,
    so a fresh checkout (or a wiped output dir) still serves it.
    """
    doc: Dict[str, Any] = {
        "instruction": entry.instruction,
        "aliases": list(entry.aliases or []),
        "objects": list(entry.objects or []),
        "priority": int(entry.priority) or 10,
        "promote": bool(entry.promoted),
        "verified": bool(entry.verified),
        "provenance": (
            f"Exported from live library entry {entry.hash} "
            f"(success={entry.success}, fail={entry.fail})."
        ),
        "score": _strip_provenance(entry.score),
    }
    if entry.notes:
        doc["notes"] = entry.notes
    return doc


def dump_seed(entry: BTEntry) -> str:
    """
    Seed YAML text for a library entry, ready to write into
    ``configs/bt_seeds/``.
    """
    return yaml.safe_dump(
        entry_to_seed(entry),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def _strip_provenance(score: Any) -> Any:
    """
    Drop planner bookkeeping (``__raw_yaml``, ``__planner``) so an exported
    seed is the tree and nothing else. bt_library already ignores these
    when hashing, so removing them does not change the entry's identity.
    """
    if isinstance(score, dict):
        return {k: _strip_provenance(v) for k, v in score.items() if not str(k).startswith("__")}
    if isinstance(score, list):
        return [_strip_provenance(v) for v in score]
    return score


def seed_slug(instruction: str) -> str:
    """
    Filename stem for a seed capturing ``instruction``.
    """
    return _norm(instruction).replace(" ", "_")[:64] or "seed"


__all__ = [
    "Seed",
    "SeedReport",
    "SeedError",
    "load_seeds",
    "parse_seed",
    "seed_library",
    "ensure_seeded",
    "entry_to_seed",
    "dump_seed",
    "seed_slug",
    "default_seed_dir",
    "DEFAULT_SEED_SUBDIR",
    "SEED_DIR_ENV",
]
