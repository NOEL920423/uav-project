# uav_navigation

Expert-flight navigation: deterministic A*, validated optional B-spline smoothing,
timed-trajectory generation, and bounded trajectory following.

The expert flight launch in `uav_px4_control` starts three nodes:
`astar_planner_node`, `trajectory_parameterizer_node`, and `trajectory_follower_node`.
BC flight evaluation does not start these nodes.

`planner/`, `trajectory/`, and `tracking/` hold the pure algorithms and ROS adapters.
`diagnostics/` retains pure geometry, kinematic-plant, and metrics fixtures used by tests.
Standalone offline launch files and embedded fixture nodes have been removed.

Scene inputs are coherent `/uav/scene/{obstacles,start,goal}` snapshots in `isaac_world`.
The planner converts them to `px4_ned`, the parameterizer publishes a validated
`/uav/trajectory/candidate`, and the follower publishes `/uav/control/astar_command`.
Only the downstream mux selects the command. This package never publishes `/fmu/in/*`.

`config/task.yaml` remains the shared goal-distance setting. Existing planner,
trajectory, follower limits, frames, timing, and acceptance rules are unchanged.
See `docs/coordinate_frames.md` and `docs/architecture.md`.
