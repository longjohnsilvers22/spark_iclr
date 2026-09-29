#!/usr/bin/env bash
# Regenerate the FR3 URDFs in this directory from upstream xacro.
#
# Run from anywhere. Requires:
#   - ROS2 Jazzy `xacro` on PATH (the spark_conda env sources it)
#   - The franka_description clone at $FRANKA_DESC (default below)
#
# The generated URDFs are post-processed to use mesh paths relative to
# this directory (../../../../../external/franka_description/...), so they
# load with yourdfpy / urdf_parser_py / pinocchio / trimesh without
# ament/ROS2 and stay portable across checkouts. For MoveIt2 usage prefer
# the `package://franka_description/...` URDF that lives inside the clone.

set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$OUT_DIR/../../../../.." && pwd)"
FRANKA_DESC="${FRANKA_DESC:-$REPO_ROOT/external/franka_description}"
REL_DESC="../../../../../external/franka_description"

if [[ ! -d "$FRANKA_DESC" ]]; then
    echo "ERROR: franka_description not found at $FRANKA_DESC" >&2
    echo "  Clone it: cd ~/spark/external && git clone --depth 1 \\" >&2
    echo "    https://github.com/frankaemika/franka_description.git" >&2
    echo "  Or:       cd ~/spark/src && vcs import ../external < ../spark_real.repos" >&2
    exit 1
fi

# Make xacro see franka_description as an ament package (no colcon needed).
mkdir -p "$FRANKA_DESC/share/ament_index/resource_index/packages"
touch "$FRANKA_DESC/share/ament_index/resource_index/packages/franka_description"
mkdir -p "$FRANKA_DESC/share/franka_description"
for d in robots end_effectors meshes accessories; do
    ln -sfn "$FRANKA_DESC/$d" "$FRANKA_DESC/share/franka_description/$d"
done
cp -f "$FRANKA_DESC/package.xml" "$FRANKA_DESC/share/franka_description/package.xml"

export AMENT_PREFIX_PATH="$FRANKA_DESC:${AMENT_PREFIX_PATH:-}"

# 1. FR3 + Franka Hand (absolute mesh paths).
cd "$FRANKA_DESC"
python scripts/create_urdf.py fr3 --abs-path --host-dir "$FRANKA_DESC"
cp -f urdfs/fr3_franka_hand.urdf "$OUT_DIR/fr3_franka_hand.urdf"
cp -f urdfs/fr3_franka_hand.srdf "$OUT_DIR/fr3_franka_hand.srdf"

# 2. FR3 bare arm (no end-effector).  create_urdf.py does not have a
#    no-EE switch that produces an "all absolute path" output, so fall
#    back to direct xacro + sed.
xacro robots/fr3/fr3.urdf.xacro hand:=false -o "$OUT_DIR/fr3_no_hand.urdf"
sed -i "s|package://franka_description|$FRANKA_DESC|g" "$OUT_DIR/fr3_no_hand.urdf"

# Make the committed URDFs portable: rewrite the absolute franka_description
# prefix to a path relative to this directory.
sed -i "s|$FRANKA_DESC|$REL_DESC|g" \
    "$OUT_DIR/fr3_franka_hand.urdf" \
    "$OUT_DIR/fr3_franka_hand.srdf" \
    "$OUT_DIR/fr3_no_hand.urdf"

echo "Wrote:"
ls -l "$OUT_DIR"/*.urdf "$OUT_DIR"/*.srdf
