# uav_data_recorder

Future owner of synchronized dataset, metadata, pose-log, and rosbag recording.

- Inputs: episode state, camera streams, vehicle state, and planning outputs.
- Outputs: versioned datasets and recording status (future).
- Must not: command PX4, generate scenes, or execute training/inference jobs.
- Phase 1 behavior: idle with recording disabled; opens no output files.
- Phase 10A: `expert_dataset_recorder` is opt-in from the guarded flight launch,
  uses bounded synchronization queues, and writes one V1 episode only under a
  Git-ignored artifact root. See `docs/phase10a_expert_dataset.md`.
- Phase 10B: batch mode keeps the BC V1 primary CSV contract unchanged, records
  optional auxiliary RGB/depth joins separately, carries seed/scene/rejection/safety
  metadata, and appends finalized episodes to the batch manifest. The guarded
  `episode_scene_client` requires landed/disarmed/no-failsafe PX4 state before
  requesting a seeded Isaac scene. See
  `docs/phase10b_multiepisode_dataset.md`.
- Phase 10C: the auxiliary RGB stream is explicitly named `observer_rgb` and
  uses the legacy Episode Manager's effective TOP geometry. The read-only
  `visual_qa_capture` saves FPV, Observer, raw FPV depth, and normalized depth
  previews at start, mid-flight, and near-goal ASTAR tracking phases. It never
  commands PX4 and is not a dataset collector. See
  `docs/phase10c_scene_camera_recovery.md`.
- Formal expert collection: the same recorder additionally publishes a 1 Hz
  file-based progress snapshot and records A* path plus raw stream counts. The
  one-command orchestration, resume manifest, aggregate validation, and 20
  episode Visual QA cadence are documented in
  `docs/expert_dataset_collection.md`. The recorder still never commands PX4.
