# uav_bringup

Owns composition and launch wiring for the tracked ROS 2 workspace.

- Inputs: launch arguments and package executables.
- Outputs: a composed ROS graph.
- Must not: hide safety policy, add PX4 publishers, or imply simulation success.
- Phase 1 behavior: `uav_system_scaffold.launch.py` starts five idle placeholders;
  every real capability is disabled and no application topic is published.
- Future phase: add staged bringup only as each package becomes implemented.
