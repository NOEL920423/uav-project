# uav_navigation

Phase 2 provides a deterministic, fixed-altitude, 8-connected A* planner. The
search, geometry, simplification, validation, and metrics modules are pure
Python. Only `astar_planner_node.py` adapts them to ROS 2.

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
- `/uav/planner/path` (`nav_msgs/msg/Path`), independently validated final path
- `/uav/planner/status` (`std_msgs/msg/String`)

All path headers and pose headers use `px4_ned` and one ROS-clock result stamp.
Identity pose orientation is a placeholder and must not be consumed as path
heading. On failure, all three durable path topics receive an explicitly empty
`px4_ned` path to clear stale results, followed by `FAILURE|reason=...`. Success
status includes method, point counts, geometric length, physical clearance, and
fallback detail. These are geometric planner metrics, not flight dynamics or
tracking claims.

## Parameters

Canonical values live in `config/astar_planner.yaml`:

| Parameter | Default | Meaning |
|---|---:|---|
| `planning_frame` | `px4_ned` | Locked Phase 2 output frame |
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
| `use_sim_time` | `false` | ROS built-in clock selection |

Parameters are read and validated at startup. Dynamic updates are intentionally
unsupported in Phase 2, so an active calculation cannot observe partial config.

## Safety and fallback

The grid planning radius is obstacle radius + 0.18 m physical radius + 0.13 m
static margin. Independent continuous validation adds another 0.07 m. RDP is
only a candidate: selection tries validated RDP, validated greedy visibility,
then the validated densified raw A* path. Failure publishes no nonempty final
path. B-spline is not implemented.

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
```

The finite harness publishes one fixed scene, validates raw/simplified/final
output and exact endpoints, checks continuous clearance and `/fmu/in/*` absence,
then exits. The planner is persistent, so the launch service terminates the
remaining planner after the required harness exits successfully.
