# uav_data_recorder

Future owner of synchronized dataset, metadata, pose-log, and rosbag recording.

- Inputs: episode state, camera streams, vehicle state, and planning outputs.
- Outputs: versioned datasets and recording status (future).
- Must not: command PX4, generate scenes, or execute training/inference jobs.
- Phase 1 behavior: idle with recording disabled; opens no output files.
- Phase 10A: `expert_dataset_recorder` is opt-in from the guarded flight launch,
  uses bounded synchronization queues, and writes one V1 episode only under a
  Git-ignored artifact root. See `docs/phase10a_expert_dataset.md`.
