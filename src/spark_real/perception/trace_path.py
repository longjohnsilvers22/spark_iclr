"""Operator-drawn 2D traces -> 3D waypoint paths for the executor.

This module back-projects the ``trace`` kind described in annotations.py;
routes/annotate_trace.py is the producer.

SPARK has no lateral motion planning: approaches are straight lines, so
nothing routes around an obstacle beside the target. A drawn path is the
operator's routing (and correction) channel.

The trace is interpreted on a HORIZONTAL PLANE at a chosen height (the target
object's z by default). A hand-drawn 2D curve cannot specify height -- that is
the one thing the operator is not asked for -- so the path stays at a single
safe height and the existing descent logic owns the vertical move.
"""

import logging

import numpy as np

from spark_real.perception.wrist_ray import pixel_to_plane_point

logger = logging.getLogger("spark_server")

# Points closer together than this add nothing but blend rows.
MIN_SPACING_M = 0.03
# The blender caps its own rows; keep the path well under that so the lift and
# descent rows it appends always fit.
MAX_WAYPOINTS = 6


def trace_to_waypoints(annotation, cal, z_plane, min_spacing_m=MIN_SPACING_M,
                       max_points=MAX_WAYPOINTS):
    """Back-project a trace onto the plane ``z = z_plane`` in the base frame.

    ``annotation`` is a ``trace``-kind Annotation (normalized x, y).
    ``cal`` is the camera's CameraCalibration (intrinsics + extrinsic).
    Returns a list of ``[x, y, z]`` in the robot base frame, decimated to at
    most ``max_points`` and never closer together than ``min_spacing_m``.
    Points whose ray misses the plane are dropped, not guessed.
    """
    pixels = annotation.to_pixels((int(cal.width), int(cal.height)))
    intr = (float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy))
    T = np.asarray(cal.extrinsic, dtype=float)

    pts = []
    missed = 0
    for u, v in pixels:
        hit = pixel_to_plane_point(u, v, intr, T, float(z_plane))
        if hit is None:
            missed += 1
            continue
        if pts and float(np.linalg.norm(np.asarray(hit)[:2] - pts[-1][:2])) < min_spacing_m:
            continue
        pts.append(np.asarray(hit, dtype=float))
    if missed:
        logger.info("[trace] %d drawn point(s) missed the z=%.3f plane", missed, z_plane)

    # Keep the ENDS and decimate the middle: the last point is where the
    # operator wants to arrive, so it must survive any thinning.
    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = [pts[i] for i in sorted(set(keep.tolist()))]
    out = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
    logger.info(
        "[trace] %d drawn point(s) -> %d waypoint(s) on the z=%.3f plane",
        len(pixels), len(out), z_plane,
    )
    return out


def waypoints_to_lead(waypoints, orientation, velocity=None, label="drawn-trace"):
    """Shape waypoints into the ``lead`` rows _transport_to already accepts.

    The transport flies ``lead`` before its own lift/approach rows as one
    blended phrase, so a drawn path costs no extra stops. The FINAL drawn
    point is deliberately dropped: it is the operator saying "go here", which
    is the target the transport already owns -- keeping it would command the
    same pose twice and fight the descent.
    """
    rows = []
    for p in list(waypoints or [])[:-1]:
        rows.append((np.asarray(p, dtype=float), list(orientation), velocity, label))
    return rows
