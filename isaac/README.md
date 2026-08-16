# Active Isaac runtime

`runtime/bootstrap.py` and `runtime/runtime_bridge.py` are the active Phase 9
Isaac/Pegasus integration. The bootstrap loads the bridge from its own
directory, so the pair must remain together.

Historical Isaac Script Editor pipelines are retained under `legacy/` and are
not part of the verified flight command.

Phase 10A optionally adds a 320x180 JPEG FPV topic without changing the default
Phase 9 runtime. Set `UAV_PHASE10A_CAMERA=1` before starting the same bootstrap.
Dataset storage and timestamp synchronization stay in the external ROS 2
recorder; the embedded bridge never writes dataset files.

Phase 10C uses the same `UAV_PHASE10B_SENSORS=1` switch, but restores the
canonical later legacy scene and camera contracts. `runtime/episode_scene.py`
generates eight decorated high-rise buildings, including two guaranteed direct
path blockers, and exact legacy episode lighting. The auxiliary camera uses the
Episode Manager's effective `TOP` Observer override and publishes on
`/uav/isaac/observer/image/compressed`. FPV uses body +X and the effective
`-0.8 m` look-down override. Its eye is applied as a rigid mount so publish-rate
world smoothing cannot put the UAV body in the FPV image. FPV remains 320x180
JPEG quality 85 and FPV depth remains uint16-millimetre PNG.

A guarded ROS client still applies scenes only from a landed, disarmed,
no-failsafe reset state. Generated sensor data remains outside Git. See
`docs/phase10c_scene_camera_recovery.md` for the exact legacy parameters and the
three-scene visual QA contract.

For formal expert collection, users do not start the sensor runtime or invoke
the scene client manually. `./uav expert-collect --episodes N` owns a fresh
Isaac/PX4/XRCE process group for each seed, enables these unchanged streams,
performs the safe scene request and flight, and cleans the group before the
next episode. See `docs/expert_dataset_collection.md`.
