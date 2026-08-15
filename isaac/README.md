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
