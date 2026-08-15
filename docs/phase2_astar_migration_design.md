# Phase 2 A* migration design

## Scope and selected sources

Phase 2 extracts deterministic, position-only A* planning from the working
legacy pipeline. It does not change or call any legacy script.

- Canonical planning functions:
  `legacy/isaac_direct_pipeline/4.px4_astar.py`, specifically
  `obstacle_planning_radius`, `obstacle_validation_radius`, grid construction,
  endpoint recovery, `astar_search_grid`, RDP/densification, greedy fallback,
  continuous segment validation, retry with segment clearance, and the 2.5D
  overflight filter.
- Cross-check: the corresponding function blocks in
  `legacy/isaac_direct_pipeline/5.astar_waypoint_exporter.py` and
  `legacy/isaac_ros2_episode_pipeline/5.astar_ros2_path_publisher.py` are byte-identical.
- Canonical obstacle geometry: the current high-rise scene generator
  `legacy/isaac_ros2_episode_pipeline/2.scene_episode_generator.py` stores a yaw-invariant
  bounding-circle radius `0.5 * hypot(width, depth)`. The Phase 2 planner trusts
  the positive finite radius supplied in `ObstacleArray`; USD/BBox extraction
  stays outside the pure planner.
- ROS message adaptation and identity path orientation are based on the
  existing ROS path publisher, but are rewritten as a normal ROS 2 node.

The comparison and function-level selections are recorded in
`astar_implementation_comparison.md`.

## Closed frame decision

The canonical planning frame for the first ROS 2 planner milestone is
`px4_ned`. Raw scene inputs arrive in `isaac_world` and are converted once at
the planner-node boundary by `coordinate_frames.py`:

```text
Isaac position [x, y, z]
  -> PX4 NED [y + offset_n, x + offset_e, -z + offset_d]
```

Offsets apply only to positions. Vectors use `[y, x, -z]` without translation.
Start and goal XY come from the converted scene poses; planning Z is fixed to
`-flight_altitude_m + ned_offset_z`, preserving legacy behavior. Obstacles are
converted from their supplied center and retain their physical radius/height.

No `map -> px4_ned` TF is introduced. `map` remains reserved for a later
deployment decision.

## Orientation boundary

The mapping proves position and free-vector conversion. For explicitly defined
planar headings, Isaac `+X/+Y` maps to NED `+Y/+X`, giving
`yaw_ned = pi/2 - yaw_isaac` after normalization when Isaac yaw is CCW from
`+X` and NED yaw is clockwise from north toward east.

The legacy source does not prove a complete body quaternion, covariance,
camera optical-axis, or TF convention. Quaternion conversion is unsupported in
Phase 2. Published `nav_msgs/Path` poses use identity orientation, matching the
legacy ROS publisher, and no Phase 2 consumer may interpret it as a path
heading.

**DECISION REQUIRED BEFORE FLIGHT INTEGRATION:** validate full vehicle-body and
quaternion transforms, origin reset behavior, camera extrinsics, and
covariance semantics against Pegasus/PX4 runtime data.

## Clock boundary

The ROS node stamps every output path from `node.get_clock().now()` and therefore
respects `use_sim_time`. Pure functions have no clock dependency. The planner is
input/event-driven and has no stateful timestamp arithmetic, so no standalone
`time_utils.py` is justified in Phase 2. Unit tests do not require `/clock`.

Wall/monotonic time is allowed only in the offline harness timeout and test
duration diagnostics. It is not written into ROS headers or used to correlate
scene inputs.

## Safety envelope

The physical formulas remain unchanged:

```text
planning_radius = obstacle.radius
                + uav_physical_radius_m
                + static_safety_margin_m

validation_radius = planning_radius
                  + minimum_segment_clearance_m
```

With canonical defaults these add `0.18 + 0.13 = 0.31 m` for occupancy and an
additional `0.07 m` for continuous validation. Grid quantization reserve is a
diagnostic only and is not mixed into the physical obstacle radius. The second
A* attempt inflates occupancy by the segment-clearance margin if the first raw
path cannot pass continuous validation.

An obstacle may be excluded from XY avoidance only when overflight is enabled
and:

```text
obstacle_top_height + overfly_vertical_clearance_m <= flight_altitude_m
```

This remains 2.5D; no vertical search is introduced.

## Planner behavior

- Deterministic 8-connected A* with Euclidean heuristic and deterministic heap
  tie breaking.
- Occupancy uses circular configuration-space obstacles at fixed altitude.
- Diagonal moves match the canonical source and are not grid-corner-blocked;
  independent continuous circle-segment validation remains mandatory before
  any final path can be returned.
- Exact endpoints are rejected when physically forbidden. If only the rounded
  endpoint cell is occupied, deterministic nearest-free-cell search is allowed
  within the configured radius and the exact endpoint is restored afterward.
- Direct-line bias and soft-clearance cost are optional, never safety overrides.
- Raw path is validated first. Simplification tries RDP plus spacing, then
  greedy safe shortcuts, then a densified validated raw fallback.
- The selected final path is validated again. No valid path produces a
  structured `PlannerResult` failure instead of an exception escaping.

Intentional corrections relative to legacy behavior:

- A path with fewer than two points is invalid rather than silently safe.
- Non-finite inputs/configuration and negative dimensions are rejected.
- Zero-length segments are rejected.
- Grid size is bounded to avoid accidental memory exhaustion.
- Failure reasons and metrics are structured rather than print-only.

## ROS node contract

Inputs use reliable/transient-local/depth-1 QoS:

- `/uav/scene/obstacles` — `uav_interfaces/msg/ObstacleArray`, `isaac_world`
- `/uav/scene/start` — `geometry_msgs/msg/PoseStamped`, `isaac_world`
- `/uav/scene/goal` — `geometry_msgs/msg/PoseStamped`, `isaac_world`

Outputs use reliable/transient-local/depth-1 QoS:

- `/uav/planner/path_raw` — `nav_msgs/msg/Path`, `px4_ned`
- `/uav/planner/path_simplified` — `nav_msgs/msg/Path`, `px4_ned`
- `/uav/planner/path` — validated `nav_msgs/msg/Path`, `px4_ned`
- `/uav/planner/status` — controlled-prefix `std_msgs/msg/String`

The node replans only after all three inputs are valid and their normalized
content differs from the previous planned snapshot. A failed replan publishes
explicit empty path messages to clear durable stale paths and a `FAILURE|...`
status. It never publishes B-spline topics or PX4 command topics.

## Excluded from Phase 2

B-spline smoothing, path following, control candidates, controller mux, PX4,
OFFBOARD/arming/takeoff, joystick, cameras, recording, Isaac/Pegasus startup,
XRCE-DDS startup, NavRL, and all flight testing are intentionally excluded.
