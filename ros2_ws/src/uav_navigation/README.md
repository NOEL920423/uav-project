# uav_navigation

Phase 3 provides a deterministic, fixed-altitude, 8-connected A* planner with
an optional clamped B-spline candidate. Search, geometry, simplification,
smoothing, validation, and metrics are pure Python. Only
`astar_planner_node.py` adapts planning to ROS 2.

This package does not follow paths, publish control commands, start PX4 or a
simulator, arm a vehicle, or publish any `/fmu/in/*` topic.

## ROS interfaces

All interfaces use reliable, transient-local, keep-last depth 1 QoS. Planning
is event-driven after a coherent, changed input snapshot; there is no periodic
replanning timer.

Inputs:

- `/uav/scene/obstacles` (`uav_interfaces/msg/ObstacleArray`), `isaac_world`
- `/uav/scene/start` (`geometry_msgs/msg/PoseStamped`), `isaac_world`
- `/uav/scene/goal` (`geometry_msgs/msg/PoseStamped`), `isaac_world`

The three input header stamps must match exactly. Position data is converted at
the node boundary to `px4_ned`; pose orientation is intentionally ignored
because full quaternion conversion is not verified. Planning Z is fixed from
`flight_altitude_m` and `ned_offset_z`.

Outputs:

- `/uav/planner/path_raw` (`nav_msgs/msg/Path`)
- `/uav/planner/path_simplified` (`nav_msgs/msg/Path`)
- `/uav/planner/path_bspline_candidate` (`nav_msgs/msg/Path`), diagnostic only
- `/uav/planner/bspline_valid` (`std_msgs/msg/Bool`)
- `/uav/planner/path` (`nav_msgs/msg/Path`), independently validated final path
- `/uav/planner/status` (`std_msgs/msg/String`)

All path headers and pose headers use `px4_ned` and one ROS-clock result stamp.
Identity pose orientation is a placeholder and must not be consumed as path
heading. On failure, all four durable path topics receive an explicitly empty
`px4_ned` path and `bspline_valid=false` to clear stale results, followed by
`FAILURE|reason=...`. Success status includes A* success, B-spline
enabled/valid/selected state, final source, point counts, geometric length,
physical clearance, curvature, rejection, and fallback detail. These are
geometric planner metrics, not flight dynamics or tracking claims.

## Parameters

Canonical values live in `config/astar_planner.yaml`:

| Parameter | Default | Meaning |
|---|---:|---|
| `planning_frame` | `px4_ned` | Locked Phase 3 output frame |
| `grid_resolution_m` | 0.05 | Grid cell size in metres |
| `grid_margin_m` | 2.0 | Dynamic grid padding in metres |
| `uav_physical_radius_m` | 0.18 | UAV horizontal radius in metres |
| `static_safety_margin_m` | 0.13 | Planning margin in metres |
| `minimum_segment_clearance_m` | 0.07 | Extra validation margin in metres |
| `endpoint_search_radius_m` | 1.0 | Nearest-free cell search radius in metres |
| `path_simplification_tolerance_m` | 0.05 | RDP tolerance in metres |
| `maximum_waypoint_spacing_m` | 1.30 | Published path spacing limit in metres |
| `use_direct_path_bias` | `true` | Enable direct-line proximity cost |
| `direct_path_bias_weight` | 0.07 | Direct-line cost weight |
| `use_clearance_aware_cost` | `true` | Enable soft clearance cost |
| `soft_clearance_radius_m` | 0.40 | Soft cost range in metres |
| `clearance_cost_weight` | 0.25 | Soft cost weight |
| `flight_altitude_m` | 2.0 | Fixed height above Isaac origin in metres |
| `enable_overfly_short_obstacles` | `true` | Enable documented 2.5D filter |
| `overfly_vertical_clearance_m` | 0.35 | Required top clearance in metres |
| `ned_offset_x/y/z` | 0.0 | Position-only NED translation in metres |
| `retry_extra_inflation_m` | 0.07 | Second-search grid inflation in metres |
| `maximum_grid_cells` | 4000000 | Grid allocation guard |
| `enable_bspline` | `true` | Generate and independently validate candidate |
| `bspline_degree` | 3 | Requested degree, reduced for short paths |
| `bspline_sample_spacing_m` | 0.08 | Arc-length resampling spacing upper bound |
| `bspline_minimum_samples` | 16 | Minimum candidate samples |
| `bspline_maximum_samples` | 1000 | Maximum candidate samples |
| `bspline_maximum_curvature` | 8.0 | Discrete curvature limit in 1/m |
| `bspline_minimum_clearance_m` | 0.07 | Candidate validation clearance |
| `bspline_preserve_endpoints` | `true` | Require exact clamped endpoints |
| `bspline_allowed_bounds_margin_m` | 0.0 | Inward explicit-bounds margin |
| `bspline_reject_self_intersection` | `true` | Reject non-adjacent 2D crossings |
| `bspline_control_point_strategy` | `validated_simplified_path` | Locked control source |
| `use_sim_time` | `false` | ROS built-in clock selection |

Parameters are read and validated at startup. Dynamic updates are intentionally
unsupported in Phase 3, so an active calculation cannot observe partial config.

## Safety and fallback

The grid planning radius is obstacle radius + 0.18 m physical radius + 0.13 m
static margin. Independent continuous validation adds another 0.07 m. RDP is
only a candidate: selection tries validated RDP, validated greedy visibility,
then the validated densified raw A* path. The B-spline uses only that validated
baseline as controls, removes adjacent duplicates, preserves exact endpoints,
resamples by approximate horizontal arc length, and independently checks
finite values, bounds, spacing/count, continuous obstacle clearance, curvature,
and self-intersection. Any rejection selects and revalidates the A* fallback.
Failure publishes no nonempty final path.

## Build and offline test

From a clean ROS 2 Jazzy shell:

```bash
cd /home/noel_614420090/uav-project/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test
colcon test-result --verbose
ros2 launch uav_navigation astar_planner_offline.launch.py
ros2 run uav_navigation geometric_path_comparison
```

The finite harness publishes one fixed scene, validates raw/simplified/final
output and exact endpoints, candidate validity/source, continuous clearance,
and `/fmu/in/*` absence, then exits. Select a fixture and state explicitly:

```bash
ros2 launch uav_navigation astar_planner_offline.launch.py \
  enable_bspline:=true fixture:=bspline-safe-single-obstacle
ros2 launch uav_navigation astar_planner_offline.launch.py \
  enable_bspline:=true fixture:=bspline-rejected-corner-cut
ros2 launch uav_navigation astar_planner_offline.launch.py \
  enable_bspline:=false fixture:=bspline-disabled
```

The planner is persistent, so the launch service terminates it after the
required finite harness exits successfully.
