# uav_data_recorder

Future owner of synchronized dataset, metadata, pose-log, and rosbag recording.

- Inputs: episode state, camera streams, vehicle state, and planning outputs.
- Outputs: versioned datasets and recording status (future).
- Must not: command PX4, generate scenes, or execute training/inference jobs.
- The recorder is idle with recording disabled and opens no output files.
- `expert_dataset_recorder` is opt-in from the guarded flight launch. It uses
  bounded synchronization queues and records FPV RGB, observer RGB, FPV depth,
  seed, scene, rejection, and safety metadata under a Git-ignored artifact root.
- `episode_scene_client` requires landed, disarmed, no-failsafe PX4 state before
  it requests a seeded Isaac scene.
- Formal expert collection: the same recorder additionally publishes a 1 Hz
  file-based progress snapshot and records A* path plus raw stream counts. The
  one-command orchestration, resume manifest, aggregate validation, and 20
  episode Visual QA cadence are documented in
  `docs/expert_dataset_collection.md`. The recorder still never commands PX4.
