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

Phase 6 adds a deterministic offline control-source multiplexer after the
Phase 5 trajectory follower. It arbitrates the canonical `HOLD`,
`ASTAR_EXPERT`, `HUMAN_JOYSTICK`, and `NAVRL_POLICY` candidates, validates the
selected command independently, and fails closed through explicit HOLD
states. It never arms, launches PX4 or Isaac Sim, loads a NavRL model, reads
joystick hardware, or publishes/remaps any `/fmu/in/*` topic.

Phase 7 adds an offline-only PX4 velocity-candidate mapping contract,
independent validator, synthetic telemetry, and fail-closed output-enable gate.
The boundary ends at the diagnostic `safe_to_forward` permission bit; it does
not start or control PX4.

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
./uav px4-map-check
./uav px4-gate-check
./uav px4-boundary-check
```

The quantitative Phase 5 tracking and Phase 6 arbitration results are in
[`docs/phase5_offline_tracking_comparison.md`](docs/phase5_offline_tracking_comparison.md)
and
[`docs/phase6_control_mux_comparison.md`](docs/phase6_control_mux_comparison.md),
with the Phase 7 boundary matrix in
[`docs/phase7_px4_boundary_validation.md`](docs/phase7_px4_boundary_validation.md).
