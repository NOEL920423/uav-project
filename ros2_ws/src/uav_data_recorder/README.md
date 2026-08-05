# uav_data_recorder

Future owner of synchronized dataset, metadata, pose-log, and rosbag recording.

- Inputs: episode state, camera streams, vehicle state, and planning outputs.
- Outputs: versioned datasets and recording status (future).
- Must not: command PX4, generate scenes, or execute training/inference jobs.
- Phase 1 behavior: idle with recording disabled; opens no output files.
- Future phase: implement explicit start/stop lifecycle and bounded queues.
