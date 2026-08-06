# Offline Trajectory Feasibility Comparison

This deterministic report uses the Phase 4 pure parameterizer and independent
validator with the checked-in conservative defaults, except the explicitly
labelled jerk-scaling fixture (`maximum_jerk_mps3=0.25`,
`maximum_yaw_rate_radps=0.3`). Coordinates are `px4_ned`; no geometry is added
or altered except removal of an adjacent duplicate.

This is not flight validation, controller validation, PX4 validation, tracking
validation, or disturbance validation. It makes no claim about real vehicle
performance.

| Path source | Points | Length m | Duration s | Max speed m/s | Max long accel m/s² | Max lateral accel m/s² | Max jerk m/s³ | Max yaw rate rad/s | Max yaw accel rad/s² | Scale | Result | Rejection reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| straight | 3 | 6.000000 | 6.000000 | 2.000000 | 0.666667 | 0.000000 | 0.222222 | 0.000000 | 0.000000 | 1.000000 | accepted | — |
| Phase 3 accepted B-spline-shaped path | 7 | 6.115771 | 4.498928 | 2.000000 | 1.500000 | 1.500000 | 2.149001 | 0.423722 | 0.526427 | 1.000000 | accepted | — |
| sharp bend | 4 | 4.000000 | 5.650276 | 1.554600 | 0.710818 | 1.500000 | 0.945563 | 0.483639 | 0.188796 | 1.000000 | accepted | — |
| high curvature | 5 | 1.069284 | 8.514253 | 0.222440 | 0.061849 | 0.196139 | 0.306217 | 1.498501 | 0.349708 | 2.765436 | accepted | — |
| adjacent duplicate | 3 | 2.000000 | 2.309401 | 1.732051 | 1.500000 | 0.000000 | 1.299038 | 0.000000 | 0.000000 | 1.000000 | accepted | duplicate removed |
| yaw wrap crossing | 4 | 3.000083 | 2.886788 | 1.732086 | 1.500000 | 0.021000 | 1.443396 | 0.006062 | 0.005833 | 1.000000 | accepted | — |
| jerk/yaw scaling | 5 | 3.000000 | 29.356096 | 0.209655 | 0.015466 | 0.048308 | 0.018612 | 0.299700 | 0.040972 | 5.572329 | accepted | — |

The high-curvature and jerk/yaw fixtures demonstrate global time scaling: the
scale changes all temporal derivatives while positions, arc length, altitude,
and curvature remain fixed. Separate harness runs cover deterministic rejection
of one-point, non-finite, wrong-frame, and insufficient-scaling-budget inputs.
