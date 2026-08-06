# UAV Isaac Sim PX4 ROS2 Project

This repository stores UAV simulation research scripts and small scene files.

## Main components

- Isaac Sim UAV simulation
- Pegasus Simulator integration
- PX4 OFFBOARD control
- A* path planning
- FPV and observer camera setup
- PNG camera recording
- ROS2 pose logging
- Manual / joystick control experiments

## Notes

Generated datasets, rosbags, logs, images, and videos are excluded from Git.
Large Isaac Sim assets should be backed up separately.

## ROS 2 developer workflow

Use the repository-local `./uav` command for isolated Jazzy build, test,
verification, interface inspection, and the non-flight offline planner harness.
See [`docs/developer_commands.md`](docs/developer_commands.md).

Phase 5 adds a deterministic, offline closed-loop trajectory-follower
candidate. It samples the validated Phase 4 timed trajectory, creates bounded
`px4_ned` velocity/yaw-rate candidates, independently validates them, and
drives only a fixed-step kinematic test plant. It never arms, launches PX4 or
Isaac Sim, or publishes/remaps any `/fmu/in/*` topic.

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
```

The quantitative eight-fixture results and their limitations are in
[`docs/phase5_offline_tracking_comparison.md`](docs/phase5_offline_tracking_comparison.md).
