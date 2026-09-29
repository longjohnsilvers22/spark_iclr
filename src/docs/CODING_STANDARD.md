# SPARK Coding Standard

One standard for all SPARK branches and machines. The deployment branches
(franka-deploy, bimanual, and others) drift per host, so consolidating them is
only tractable if every branch is written to the same rules. This file is that
reference. It is enforced by tooling, not just convention (see "Enforcement").

## Style rules

These apply to code, comments, docstrings, string literals, log lines, and
printed output. They do not apply to LaTeX under `paper/`.

- No em dashes (U+2014) or en dashes (U+2013). Use a comma, a colon, parentheses,
  or two sentences. A hyphen between words is fine.
- No decorative divider lines. Do not print or write runs of repeated characters
  as separators (no `====`, `----`, `****`, `####`). Use a short plain label.
- No single-line docstrings. For a one-line description use a `#` comment, not a
  triple-quoted string. Use a docstring only when the documentation genuinely
  spans multiple lines, for example the arguments and returns of a public API.
- Run ruff and black. Do not hand align code or comments.
- Names describe intent. Avoid abbreviation soup and single-letter names outside
  short loops or math.

## Structure rules

- Reuse packages. Before writing new code, check the project dependencies and
  existing helpers. Prefer numpy, scipy, transforms3d, opencv, and mujoco over
  reimplementing geometry, IO, parsing, or math.
- All rotation and quaternion math goes through `scipy.spatial.transform.Rotation`
  (or transforms3d). Do not hand roll quaternion conversion, axis-angle, or
  rotation-matrix code.
- One implementation per concept. There must be a single source of truth for IK,
  for transforms, and for each control primitive. New call sites import it rather
  than copying it.
- Keep modules focused. A file over roughly 800 lines is a smell; split by
  responsibility. A function that does not fit on a screen is a smell; extract.
- Surgical changes. Touch only what the task requires, match the surrounding
  style, and do not delete code you did not write without confirming it is unused.

## Canonical modules (single source of truth)

Do not reimplement these. Import them.

- Real-robot IK: `spark_real/control/fr3_ik_pyroki.py` (pyroki, default) with the
  pinocchio fallback in `spark_real/control/fr3_ik.py`.
- Sim and benchmark IK: `spark_bench/libero_pro/ik.py` (compute_ik,
  compute_ik_6dof, solve_ik_sideways) on top of `spark_bench/pyroki_ik.py`
  (solve_ik_6dof, quat_from_approach).
- Transforms: `scipy.spatial.transform.Rotation`.

## Enforcement

- Run `ruff` and `black` before committing; both are configured for the repo.
- The style and structure rules above are checked in review.
