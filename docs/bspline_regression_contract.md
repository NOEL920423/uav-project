# B-spline deterministic regression contract

## Pure basis and evaluation

- Open-uniform knot vectors are finite, nondecreasing, and have the required
  multiplicity at both clamped endpoints.
- Basis values form a partition of unity on the closed parameter interval.
- De Boor evaluation returns the exact first and last controls.
- Repeated calls with identical inputs/configuration return identical results.
- Invalid degree, knots, controls, and non-finite intermediate values fail with
  structured diagnostics rather than escaping into ROS publication.

## Short paths and endpoint preservation

- Fewer than two distinct controls fails.
- Two controls use degree 1; three use at most degree 2; four or more use
  `min(configured degree, point_count - 1)`.
- Adjacent duplicates are removed deterministically; nearly coincident controls
  use the planner numerical tolerance.
- Candidate start and goal equal the validated A* endpoints within strict
  tolerance, and every point keeps the fixed `px4_ned` altitude.

## Spatial resampling

- Provisional evaluation precedes cumulative arc-length interpolation.
- Output spacing is approximately uniform and never exceeds the configured
  limit plus numerical tolerance.
- Cumulative progress is finite and monotonic, endpoints are exact, and final
  sample count remains within configured bounds.
- Zero-length input/provisional segments are removed or rejected explicitly;
  an insufficient maximum sample count rejects the candidate.

## Independent validation

- Open-space curves pass.
- Obstacle-cutting and below-clearance curves fail continuous segment checks.
- Non-finite, endpoint, altitude, bounds, spacing, zero-length, curvature, and
  self-intersection failures have deterministic reasons.
- The validation radius is never weaker than the Phase 2 physical plus static
  plus continuous-clearance envelope.
- A self-intersection identifies the two non-adjacent segment indices; adjacent
  shared endpoints do not fail.

## Selection and fallback

- Disabled B-spline selects `ASTAR_SIMPLIFIED` and reports `disabled`.
- A valid candidate selects `BSPLINE` and sets valid/selected true.
- Any rejected candidate selects the already validated baseline as
  `ASTAR_FALLBACK`, preserves overall success, and exposes its reason.
- No valid A* baseline produces overall failure and no nonempty final path.
- Candidate rejection never triggers another A* search.

## Geometric metrics

- A straight line has near-zero curvature.
- A fixed circular arc approximates its known reciprocal-radius curvature.
- Length, point count, physical clearance, segment length, heading-change, and
  curvature mean/maximum/variance are deterministic.
- Reports are labeled `Geometric Path Comparison`; no tracking, flight-time,
  collision-rate, acceleration, jerk, or dynamic-feasibility claim is allowed.

## ROS offline fixtures

The finite harness accepts a named deterministic fixture and fails nonzero for
an unexpected candidate state, source, frame, endpoint, final validation, or
forbidden topic. The selectable fixture vocabulary is:

- `bspline-safe-open-space`
- `bspline-safe-single-obstacle`
- `bspline-rejected-corner-cut`
- `bspline-rejected-clearance`
- `bspline-disabled`
- `short-two-point-path`
- `three-point-path`
- `duplicate-control-point-path`
- `self-intersection-candidate`
- `curvature-limit-rejection`

Unit tests own direct control-point edge cases. ROS fixtures own publication,
status, fallback, frame, endpoint, continuous final-validation, and
`/fmu/in/*` graph assertions. Fixed seeds are required for any randomized finite
input coverage.
