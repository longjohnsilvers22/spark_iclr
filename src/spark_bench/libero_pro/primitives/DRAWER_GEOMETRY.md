# Drawer handle geometry, measured 2026-08-26

Measured from the LIBERO asset itself, not inferred from behaviour:
`~/.cache/libero/assets/articulated_objects/wooden_cabinet.xml`, loaded in
MuJoCo, world-frame extents computed as `|R| @ half_size` so the box
quaternions are honoured. Reproduce with the snippet at the end.

## Numbers

Handle bar, identical on all three drawers: 89 mm long, 15.4 mm deep,
16.4 mm tall, centred at y = -0.1016 (the front-most structure on the
cabinet).

| Drawer | Bar z range | Clearance above bar | Clearance below bar |
|---|---|---|---|
| top | 0.1757 to 0.1921 | unobstructed | 28 mm |
| middle | 0.1022 to 0.1186 | 57 mm (bar above) | 23 mm |
| bottom | 0.0329 to 0.0493 | 53 mm (bar above) | n/a |

Bar-to-face gap, which is the slot the top-down rear finger has to enter:
**16.0 mm**.

## What this says about the two approaches

The current primitive descends vertically and pinches the bar, with the jaws
separating along the pull axis. That is a sound grasp and the docstring
explains why. Its problem is not the fingertips. The 16 mm bar-to-face gap
minus finger width is what leaves the roughly +/-3 mm of lateral tolerance the
primitive already documents, against roughly 5 mm of handle-detection noise.
The tolerance is smaller than the noise, so the sweep-and-commit search in
`handle_open_drawer` is doing the only thing available to it.

The occlusion is separate and worse. To pinch the middle bar from above, the
fingertips reach z = 0.1104 and the gripper BODY sits five to ten centimetres
higher, which puts it at z = 0.16 to 0.21. The top drawer's bar occupies
0.1757 to 0.1921 and its face bottom is at 0.1488. The hand collides even
though the fingers would fit in the 57 mm gap. That is the "shadowed by the
bar above them" limitation, stated as a measurement.

A frontal approach changes both terms. The hand body stays at y < -0.11, in
free space ahead of the cabinet, so nothing shadows it at any drawer level.
The fingers straddle the bar vertically instead of entering the bar-to-face
slot, so the binding tolerance becomes the 57 mm above and 23 mm below rather
than 16 mm minus finger width. That is roughly +/-10 mm against the same 5 mm
of detection noise, which inverts the tolerance-to-noise ratio that currently
defeats the middle and bottom drawers.

## What is NOT established

Whether the arm can reach the frontal pose. The cabinet sits against the
workspace edge in several LIBERO layouts and the approach direction is
perception-derived, so reachability has to be checked per task before any
success claim. The +/-10 mm figure is a geometric bound, not a measured
success rate. Iterations it-26, it-27, it-29 and it-30 recorded in
`drawer.py` show this primitive resists tidy fixes, and nothing here
contradicts that: the claim is that the frontal approach removes the
occlusion and relaxes the tolerance, not that it makes the cells pass.

## Reproduce

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("wooden_cabinet.xml")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for i in range(m.ngeom):
    if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
        continue
    R = d.geom_xmat[i].reshape(3, 3)
    ext = np.abs(R) @ m.geom_size[i]
    print(d.geom_xpos[i], ext)
```

## REFUTED, 2026-08-26: the frontal approach is not reachable

The geometry above is correct and the reasoning from it was wrong. Measured in
the real libero_goal middle-drawer scene, with the cabinet_middle body at
[0.023, -0.247, 0.905] and the EE starting at [-0.208, 0.000, 1.173]:

| Wrist target | IK convergence, standoff | IK convergence, bar |
|---|---|---|
| top-down, -Z palm | 2.8 mm | 2.8 mm |
| frontal, +Y palm | 169.7 mm | 132.6 mm |

Same targets, same damped-least-squares solver, only the wrist orientation
differs. The solver reaches the top-down pose to within 3 mm and misses the
frontal pose by 13 to 17 cm. This is not a controller or step-budgetproblem: a
convergence sweep at kp=40/kd=12 over 200 steps, kp=120/kd=25 over 250, and
kp=120/kd=25 over 400 left 439, 353 and 370 mm of error respectively, which is
the signature of driving toward a joint configuration that does not put the EE
where it was asked to go.

So the frontal approach cannot be commanded on this arm at this cabinet, and
the top-down bar pinch in handle_open_drawer is not a compromise. It is the
only orientation the Panda can achieve at the handle. The clearance analysis
above still explains WHY the middle and bottom cells fail, but it does not
point to a fix that this embodiment can execute.

What remains true and still unexploited: the binding constraint on the
top-down path is a 16 mm slot against roughly 5 mm of detection noise. That is
a PERCEPTION problem, not a pose problem, so the levers worth trying next are
ones that reduce handle-localisation error (multi-view handle detection,
wrist-camera refinement at the standoff, or a contact-based search that uses
the arm itself as the sensor) rather than ones that change the approach angle.

The frontal code path is left in place because it fails safe: it probes, logs
the miss, and falls through to the unchanged top-down sweep.
SPARK_DRAWER_FRONTAL=0 removes it entirely.
