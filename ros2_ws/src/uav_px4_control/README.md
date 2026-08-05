# uav_px4_control

Future sole owner of validated PX4 offboard command publication and telemetry
adaptation.

- Inputs: selected candidate command, mission state, and PX4 telemetry (future).
- Outputs: PX4 input topics and normalized vehicle state (future).
- Must not: plan paths, capture cameras, or start training jobs.
- Phase 1 behavior: idle, `enable_px4_output=false`, no `px4_msgs` dependency,
  no publishers, and specifically no `/fmu/in/*` ownership.
- Future phase: add safety gates before any PX4 publisher is introduced.
