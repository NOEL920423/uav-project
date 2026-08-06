# Phase 3 B-spline design

## Scope and safety boundary

Phase 3 adds an optional geometric B-spline candidate after the already
validated simplified A* path. It remains a non-flight planning feature: it does
not start a simulator, publish control commands, interact with PX4, or publish
any `/fmu/in/*` topic. The validated A* result remains the safety baseline and
is never discarded merely because smoothing was requested.

The selection pipeline is:

```text
raw A*
  -> safe simplification
  -> independently validated simplified A*
  -> optional B-spline candidate
  -> dense provisional evaluation
  -> spatial arc-length resampling
  -> independent B-spline validation
  -> validated B-spline or validated A* fallback
```

No candidate becomes `/uav/planner/path` unless every applicable validation
gate passes. Publishing the diagnostic candidate does not assert that it is
safe.

## Input and configuration contract

The smoother consumes the validated simplified A* path in `px4_ned`. At least
two distinct finite points are required. Adjacent points closer than the
planner numerical tolerance are removed deterministically. Removing duplicates
must still leave two points; otherwise candidate generation fails structurally.

`BSplineConfig` contains the enable flag, requested degree, spatial sample
spacing, minimum and maximum samples, maximum curvature, minimum continuous
clearance, endpoint policy, bounds margin, self-intersection policy, and control
point strategy. The only production strategy is `validated_simplified_path`.
The B-spline clearance is interpreted as the continuous segment-clearance term
in the existing validation-radius formula and is clamped to be no weaker than
`PlannerConfig.minimum_segment_clearance_m`.

Defaults use degree 3, 0.08 m maximum spatial spacing, 16 to 1000 final samples,
0.07 m continuous clearance, exact endpoints, self-intersection rejection, and
a maximum discrete curvature of 8.0 1/m. The curvature limit corresponds to a
geometric radius of 0.125 m; it is only a conservative shape gate and is not a
claim of dynamically feasible flight.

## Mathematical definition and degree behavior

For control points `P[0] ... P[n]` and effective degree `p`, the implementation
uses a clamped open-uniform knot vector with `p + 1` zeros, uniformly spaced
interior knots, and `p + 1` ones. Curve points are evaluated deterministically
with De Boor recursion. The basis is also exposed for partition-of-unity tests.

The effective degree is `min(requested_degree, point_count - 1)`. Two points use
degree 1, three points use at most degree 2, and four or more use the configured
degree subject to that limit. A single point fails. The clamped knot vector
mathematically interpolates the first and last controls; exact endpoint equality
is checked again after resampling.

## Endpoint, altitude, and resampling strategy

The provisional curve is evaluated more densely than the requested output
spacing. Cumulative 2D arc length must be finite and strictly progress after
duplicate removal. Uniform target distances are then interpolated along the
provisional polyline. The final sample count is bounded by the configured
minimum and maximum. If the maximum count cannot satisfy the maximum spacing,
the candidate is rejected instead of silently violating the spacing contract.

The first and last values come from the original validated controls exactly.
The complete post-resampling path is revalidated, so endpoint replacement can
never introduce an unchecked segment. All current A* controls use one fixed
planning altitude; every candidate point must match it within the planner
numerical tolerance. No translation offset is applied by the smoother.

## Independent safety validation order

Validation is deliberately independent of B-spline generation and runs in this
order:

1. finite values and at least two points;
2. exact requested start and goal;
3. fixed planning altitude;
4. optional inward planning-bounds margin;
5. nonzero segments and maximum spatial spacing;
6. every segment against every obstacle validation radius;
7. configured B-spline minimum clearance, never weaker than Phase 2;
8. discrete curvature limit;
9. optional 2D polyline self-intersection rejection.

Obstacle validation uses:

```text
obstacle radius
  + UAV physical radius
  + static safety margin
  + max(existing minimum segment clearance, B-spline minimum clearance)
```

## Curvature and self-intersection

Curvature is estimated for each consecutive non-degenerate triple using
circumcircle curvature `4 * triangle_area / (a * b * c)`, in 1/m. Collinear or
numerically tiny-area triples produce zero curvature; zero-length legs are
reported before curvature is evaluated. Mean, maximum, and population variance
are geometric comparison metrics only.

Self-intersection uses deterministic orientation and on-segment tests on all
non-adjacent 2D segment pairs. Adjacent segments, including the first and last
segments of an open polyline, are not treated as intersections merely for
sharing an endpoint. The numerical tolerance comes from `PlannerConfig`; a
rejection reports both segment indices.

## Selection and fallback

Disabled smoothing selects the validated simplified A* path with source
`ASTAR_SIMPLIFIED`. An accepted candidate selects source `BSPLINE`. Any candidate
generation or validation rejection selects the existing validated A* path with
source `ASTAR_FALLBACK` and a structured rejection reason. B-spline rejection is
not an overall planning failure. If the A* baseline itself is invalid, planning
fails and no nonempty final path is published; A* is not rerun because a spline
was rejected.

## ROS publication semantics

On successful A* planning the node publishes, in order, raw A*, validated
simplified A*, the B-spline candidate when generated, `bspline_valid`, the final
validated path, and structured status. All path messages use `px4_ned`:

- `path_raw`: diagnostic grid result;
- `path_simplified`: validated safety baseline;
- `path_bspline_candidate`: untrusted diagnostic candidate;
- `path`: trusted selected result;
- `bspline_valid`: true only after every B-spline gate passes.

Recommended RViz colors are gray for raw, blue for simplified, orange for the
candidate, and green for final. These are visualization semantics only; Phase 3
does not add Isaac visualization.

## Intentionally excluded behavior and limitations

Phase 3 adds no follower, tracking, command generation, acceleration/jerk
analysis, OFFBOARD behavior, arming, takeoff, landing, simulator startup, camera,
recorder, NavRL, or flight experiment. Geometric length, heading change,
clearance, and curvature do not establish vehicle dynamic feasibility. Body and
quaternion conventions, controller constraints, timing, and closed-loop safety
remain decisions for later phases.
