# uav_interfaces

Phase 8 adds `Px4StreamStatus.msg` and `SetPx4StreamEnable.srv` for the
separately gated SITL-only setpoint streamer. These interfaces do not authorize
OFFBOARD, arming, takeoff, landing, or real-vehicle operation.

Owns the typed ROS 2 message, service, and action contracts shared by the UAV
integration packages. Current definitions include scene/episode contracts,
`TimedTrajectory`, `TrajectoryPoint`, `TrajectoryTrackingStatus`, the Phase 6
`ControlMuxStatus`, and `SetControlSource` service. Runtime behavior remains in
the owning packages.

- Inputs: interface definitions maintained in this package.
- Outputs: generated language bindings after a workspace build.
- Must not: contain planners, controllers, PX4 command logic, or simulator I/O.
- Phase 6: `uav_px4_control/control_mux` owns the mux status publisher and
  source-selection service; this package contains definitions only.
