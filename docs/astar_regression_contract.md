# A* deterministic regression contract

## Purpose

These tests lock safety and externally meaningful behavior without depending on
fragile full grid-cell sequences. All fixtures are pure Python unless marked as
the offline ROS integration fixture.

## Coordinate fixtures

- Origin and `+X/+Y/+Z` basis mapping.
- Exact inverse and fixed-seed finite-point round trips.
- Nonzero translation offsets on positions only.
- Velocity/acceleration vectors never receive translation.
- Planar heading and yaw conversions use explicit conventions.
- Non-finite values fail; quaternion conversion remains unsupported.

## Safety-envelope fixtures

- Planning radius equals obstacle radius plus 0.18 m physical radius plus
  0.13 m static margin by default.
- Validation radius adds exactly 0.07 m.
- Point and segment clearance signs are checked at outside, tangent, and inside
  positions.
- Grid quantization reserve does not change either physical formula.
- Overflight tests cover clearly short, exact threshold, slightly tall,
  disabled mode, and negative/non-finite height rejection.

## Planner fixtures

1. No obstacles yields a safe direct final path.
2. One direct blocker yields a path around the validation envelope.
3. A gap wider than two validation radii remains passable.
4. A narrower gap is rejected or routed around, never crossed.
5. A near-obstacle start outside the validation envelope is valid.
6. A start inside the planning/validation forbidden region fails structurally.
7. A forbidden goal fails structurally.
8. A short obstacle is filtered and the direct path remains available.
9. A tall obstacle is retained and avoided.
10. A complete barrier within explicit allowed bounds yields `no path`.
11. Successful raw, simplified, and final paths preserve exact endpoints.
12. An unsafe RDP shortcut is rejected in favor of a safe fallback.
13. The fallback selector accepts a validated raw path when supplied
    simplification candidates are unsafe.
14. Repeated identical inputs and configuration produce identical results.

Assertions target success/failure, endpoint equality, continuous clearance,
side/route properties, bounded path length, deterministic equality, fallback
reason, and structured diagnostics. Exact full grid sequences are asserted only
for repeated identical runs.

## Validation and metrics contract

- Every input point must be finite and a path must contain two distinct
  endpoints.
- Expected start and goal must match exactly within the configured numerical
  tolerance.
- Every segment is checked against every validation radius.
- An error names both segment index and obstacle when collision occurs.
- Optional planning bounds and maximum waypoint spacing are validated.
- Metrics include point counts, 2D path length, physical obstacle clearance,
  mean/max segment length, and mean/max/variance of absolute heading changes.
- Geometric metrics must not be labeled as flight smoothness, acceleration,
  jerk, or tracking performance.

## ROS offline fixture

The offline harness publishes a fixed `isaac_world` obstacle/start/goal scene,
waits for raw/simplified/final paths and success status, then verifies:

- all three paths were received;
- final frame is `px4_ned`;
- exact converted start/goal are preserved;
- continuous validation succeeds;
- no `/fmu/in/*` topic exists.

It uses no Isaac, Pegasus, PX4, XRCE-DDS, camera, recorder, or controller
process. A launch timeout may stop the persistent planner after the harness
exits and must be reported as an expected timeout rather than a test failure.
