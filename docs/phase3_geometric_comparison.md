# Phase 3 Geometric Path Comparison

Date: 2026-08-06 (Asia/Taipei)

This is a deterministic geometric comparison only; it is not a
flight-performance or dynamic-feasibility comparison. The A* column is the
independently validated simplified A* baseline. Candidate metrics are retained
even when a safety gate rejects the B-spline and selects the A* fallback.

Reproduce after building and sourcing the workspace:

```bash
ros2 run uav_navigation geometric_path_comparison
```

| Scene | Candidate | Rejection / fallback reason | A* length [m] | B-spline length [m] | A* clearance [m] | B-spline clearance [m] | A* max heading [rad] | B-spline max heading [rad] | A* heading variance [rad^2] | B-spline heading variance [rad^2] | A* max curvature [1/m] | B-spline max curvature [1/m] | Final source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| open-straight | accepted | none | 4.000000 | 4.000000 | inf | inf | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | BSPLINE |
| open-diagonal | accepted | none | 4.176123 | 4.176123 | inf | inf | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | BSPLINE |
| single-obstacle | accepted | none | 4.522287 | 4.345909 | 0.460000 | 0.440889 | 0.785398 | 0.074905 | 0.092222 | 0.000469 | 1.121493 | 0.947888 | BSPLINE |
| large-obstacle-detour | accepted | none | 7.214098 | 6.959141 | 0.510000 | 0.509986 | 0.769006 | 0.069537 | 0.141744 | 0.000441 | 0.869923 | 0.869260 | BSPLINE |
| strict-clearance-gate | rejected | segment 16 intersects obstacle center | 4.522287 | 4.345909 | 0.460000 | 0.440889 | 0.785398 | 0.074905 | 0.092222 | 0.000469 | 1.121493 | 0.947888 | ASTAR_FALLBACK |
| strict-curvature-gate | rejected | candidate exceeds maximum curvature | 4.522287 | 4.345909 | 0.460000 | 0.440889 | 0.785398 | 0.074905 | 0.092222 | 0.000469 | 1.121493 | 0.947888 | ASTAR_FALLBACK |

## Complete metric set

| Scene/path | Points | Length [m] | Min clearance [m] | Mean segment [m] | Max segment [m] | Mean heading [rad] | Max heading [rad] | Heading variance [rad^2] | Mean curvature [1/m] | Max curvature [1/m] | Curvature variance [1/m^2] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| open-straight/A* | 5 | 4.000000 | inf | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| open-straight/B-spline | 51 | 4.000000 | inf | 0.080000 | 0.080000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| open-diagonal/A* | 5 | 4.176123 | inf | 1.044031 | 1.044031 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| open-diagonal/B-spline | 54 | 4.176123 | inf | 0.078795 | 0.078795 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| single-obstacle/A* | 7 | 4.522287 | 0.460000 | 0.753715 | 1.250000 | 0.596722 | 0.785398 | 0.092222 | 0.769554 | 1.121493 | 0.160845 |
| single-obstacle/B-spline | 56 | 4.345909 | 0.440889 | 0.079017 | 0.079023 | 0.036451 | 0.074905 | 0.000469 | 0.461269 | 0.947888 | 0.075106 |
| large-obstacle-detour/A* | 9 | 7.214098 | 0.510000 | 0.901762 | 1.078483 | 0.434671 | 0.769006 | 0.141744 | 0.470585 | 0.869923 | 0.167318 |
| large-obstacle-detour/B-spline | 88 | 6.959141 | 0.509986 | 0.079990 | 0.079996 | 0.034711 | 0.069537 | 0.000441 | 0.433913 | 0.869260 | 0.068868 |
| strict-clearance-gate/A* | 7 | 4.522287 | 0.460000 | 0.753715 | 1.250000 | 0.596722 | 0.785398 | 0.092222 | 0.769554 | 1.121493 | 0.160845 |
| strict-clearance-gate/B-spline | 56 | 4.345909 | 0.440889 | 0.079017 | 0.079023 | 0.036451 | 0.074905 | 0.000469 | 0.461269 | 0.947888 | 0.075106 |
| strict-curvature-gate/A* | 7 | 4.522287 | 0.460000 | 0.753715 | 1.250000 | 0.596722 | 0.785398 | 0.092222 | 0.769554 | 1.121493 | 0.160845 |
| strict-curvature-gate/B-spline | 56 | 4.345909 | 0.440889 | 0.079017 | 0.079023 | 0.036451 | 0.074905 | 0.000469 | 0.461269 | 0.947888 | 0.075106 |

## Interpretation and limitations

The accepted obstacle scenes show smaller discrete heading changes after
arc-length resampling, but this does not prove trackability. Curvature is a
three-point circumcircle estimate in the horizontal plane. The comparison does
not model time parameterization, velocity, acceleration, jerk, vehicle attitude,
controller bandwidth, disturbances, or simulator/flight behavior. Those remain
unresolved dynamic-feasibility risks outside Phase 3.
