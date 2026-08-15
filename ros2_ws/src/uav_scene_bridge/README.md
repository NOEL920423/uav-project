# uav_scene_bridge

Fail-closed Phase 9 adapter between Isaac's embedded Python runtime and the
typed ROS 2 planning boundary. It remains disabled unless
`enable_scene_access:=true` is explicitly supplied.

- Inputs: `/isaac_uav/pose` and `/uav/isaac/runtime_status`.
- Outputs: coherent transient-local `/uav/isaac/scene/{obstacles,start,goal}`
  plus live `/uav/isaac/bridge_status` health.
- Validation: documented frame, finite pose/scene values, increasing heartbeat
  sequence, stage/timeline readiness, and bounded pose/status age.
- It does not plan, command PX4, access cameras, or record datasets.
