# Legacy and prototype code

Nothing in this directory is part of the supported Phase 9 flight runtime.
Files are retained without deduplication for provenance and possible future
reference.

- `isaac_direct_pipeline/`: direct-pymavlink Isaac Script Editor pipeline.
- `isaac_ros2_episode_pipeline/`: superseded builtins/runpy ROS-in-Isaac flow.
- `pipeline/`: obsolete launch wrapper and its historical BC instructions.
- `wrappers/`: small wrappers for the direct pipeline, including one known
  broken recorder wrapper retained as evidence.
- `docs/`: historical architecture descriptions.

Do not use these files as current startup instructions.
