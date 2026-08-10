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

## Behavior Cloning ML Phase 1

The ROS-independent BC v0 pipeline lives in `uav_ml/`. It uses a 64x64 depth
observation plus vehicle velocity and goal direction to imitate the existing
bounded A*/B-spline trajectory follower command. Generated datasets,
checkpoints, and training runs are not committed.

```bash
./uav ml-doctor
./uav dataset-generate-synthetic --output datasets/bc_v0 --episodes 8
./uav dataset-check --dataset datasets/bc_v0
./uav bc-smoke-test --device cpu
./uav bc-train --dataset datasets/bc_v0 --epochs 10 --device auto
./uav ml-test
```

The synthetic generator is only a software fixture. See
`docs/ml_training_architecture.md` and `docs/ml_phase1_results.md` for the real
Isaac bridge boundary and current evidence.

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
