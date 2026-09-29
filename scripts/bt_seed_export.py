#!/usr/bin/env python3
"""
Freeze a proven behaviour tree out of the live BT library into a
version-controlled seed file.

A verified tree lives only in the runtime library under the output
directory. Exporting it to ``src/spark_real/configs/bt_seeds/<slug>.yaml``
makes it part of the repo, so every checkout serves the task from cache with
no LLM call.

Offline: reads and writes files only. Never contacts the robot, a camera,
or an LLM.

Usage::

    # what is in the library, and which entries are worth freezing
    python scripts/bt_seed_export.py --list

    # freeze the best entry for a task (writes configs/bt_seeds/<slug>.yaml)
    python scripts/bt_seed_export.py --instruction "put the pen in the bin"

    # freeze a specific tree by hash, print instead of writing
    python scripts/bt_seed_export.py --hash 0b20251e4902 --stdout

    # freeze every promoted entry in one go
    python scripts/bt_seed_export.py --all-promoted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from spark_real.bt_library import BTLibrary  # noqa: E402
from spark_real.pipeline_types import PipelineConfig  # noqa: E402
from spark_real.planning.bt_seeds import (  # noqa: E402
    default_seed_dir,
    dump_seed,
    seed_slug,
)


def default_library_dir() -> Path:
    """
    Where the runtime library lives by default: ``<output_dir>/../bt_library``
    relative to the package's configured output directory.
    """
    return Path(PipelineConfig.output_dir).parent / "bt_library"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--library",
        type=Path,
        default=None,
        help="library directory (default: <output_dir>/../bt_library)",
    )
    ap.add_argument(
        "--seed-dir",
        type=Path,
        default=None,
        help="where to write (default: the packaged configs/bt_seeds)",
    )
    ap.add_argument("--instruction", help="export the best entry for this task")
    ap.add_argument("--hash", help="export this exact entry")
    ap.add_argument(
        "--all-promoted",
        action="store_true",
        help="export every promoted entry",
    )
    ap.add_argument("--list", action="store_true", help="list entries and exit")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing seed file",
    )
    args = ap.parse_args(argv)

    lib_dir = args.library or default_library_dir()
    if not lib_dir.exists():
        print(f"no library at {lib_dir}", file=sys.stderr)
        return 2
    lib = BTLibrary(lib_dir)
    seed_dir = Path(args.seed_dir) if args.seed_dir else default_seed_dir()

    if args.list or not (args.instruction or args.hash or args.all_promoted):
        print(f"library: {lib_dir}  ({len(lib)} entries)")
        print(f"seeds:   {seed_dir}")
        print()
        rows = sorted(
            lib.entries(),
            key=lambda e: (-e.success, e.fail, e.instruction),
        )
        for e in rows:
            flag = "S" if e.seed else ("P" if e.promoted else " ")
            print(f"  {flag} {e.hash}  ok={e.success:<3} fail={e.fail:<3} " f"{e.instruction!r}")
        if not args.list:
            print("\nnothing to export; pass --instruction / --hash / --all-promoted")
        return 0

    targets = []
    if args.hash:
        entry = lib.get(args.hash)
        if entry is None:
            print(f"no such BT: {args.hash}", file=sys.stderr)
            return 2
        targets.append(entry)
    if args.instruction:
        match = lib.lookup(args.instruction, min_success=0)
        if match is None:
            print(
                f"no cached BT for {args.instruction!r}. Run the task once "
                f"with the planner on, then export it.",
                file=sys.stderr,
            )
            return 2
        targets.append(match.entry)
    if args.all_promoted:
        targets.extend(e for e in lib.entries() if e.promoted and not e.seed)

    seen = set()
    written = 0
    for entry in targets:
        if entry.hash in seen:
            continue
        seen.add(entry.hash)
        text = dump_seed(entry)
        if args.stdout:
            print(f"# --- {entry.hash} ---")
            print(text)
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        path = seed_dir / f"{seed_slug(entry.instruction)}.yaml"
        if path.exists() and not args.force:
            print(f"exists, skipping (use --force): {path}", file=sys.stderr)
            continue
        path.write_text(text)
        print(f"wrote {path}  <- {entry.hash} ({entry.instruction!r})")
        written += 1

    if written:
        print(
            f"\n{written} seed(s) written. They load on the next server start, "
            f"or POST /api/bt/reload_seeds now."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
