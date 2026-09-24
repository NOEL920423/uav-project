# Active Isaac runtime

`runtime/bootstrap.py` and `runtime/environment.py` are the active
Isaac/Pegasus integration. The bootstrap loads the environment module from its
own directory, so the pair must remain together.

Historical Isaac Script Editor pipelines are retained under `legacy/` and are
not part of the verified flight command.

Set `UAV_FPV_CAMERA=1` to enable the optional 320x180 JPEG FPV topic without
changing the default runtime. Dataset storage and timestamp synchronization
stay in the external ROS 2 recorder; the embedded bridge never writes dataset
files.

Set `UAV_EXPERT_SENSORS=1` for the canonical expert camera contracts.
`runtime/environment.py`
generates eight decorated high-rise buildings, including two guaranteed direct
path blockers, and exact legacy episode lighting. The auxiliary camera uses the
Episode Manager's effective `TOP` Observer override and publishes on
`/uav/isaac/observer/image/compressed`. FPV uses body +X and the effective
`-0.8 m` look-down override. Its eye is applied as a rigid mount so publish-rate
world smoothing cannot put the UAV body in the FPV image. FPV remains 320x180
JPEG quality 85 and FPV depth remains uint16-millimetre PNG.

A guarded ROS client still applies scenes only from a landed, disarmed,
no-failsafe reset state. Generated sensor data remains outside Git.

For formal expert collection, users do not start the sensor runtime or invoke
the scene client manually. `./uav expert-collect --episodes N` owns a fresh
Isaac/PX4/XRCE process group for each seed, enables these unchanged streams,
performs the safe scene request and flight, and cleans the group before the
next episode. See `docs/expert_dataset_collection.md`.
