# Expert Dataset Collection Tool

The formal collector prepares a resumable, one-command dataset workflow around
the already verified Isaac → ROS 2 → A* → PX4 flight pipeline. Its dataset is:

```text
artifacts/datasets/bc_expert_highrise_v1/
```

The entire `artifacts/` generated-data tree is Git-ignored. Images, depth PNGs,
flight evidence, manifests, runtime logs, and contact sheets must not be
committed.

## Commands

Collect 100 accepted, validated successful episodes into a new dataset:

```bash
./uav expert-collect --episodes 100
```

If that run is interrupted, resume toward the same dataset-wide accepted target:

```bash
./uav expert-collect --episodes 100 --resume
```

`--episodes` is the dataset-wide number of accepted successful episodes, not
the number of seed attempts. Existing accepted episodes count toward the target.
For example, if a dataset already has 8 accepted episodes, this collects 92 more:

```bash
./uav expert-collect --episodes 100 --resume
```

Each rejected attempt advances to the next canonical seed and remains in the
append-only attempt history. Collection stops when the accepted target is met
or the total attempt limit is reached. The default limit is
`ceil(1.5 * --episodes)`; it can be set explicitly:

```bash
./uav expert-collect --episodes 100 --max-attempts 150 --resume
```

Completed episode directories are never overwritten. Any incomplete directory
is moved to the Git-ignored recovery log before that interrupted attempt is
resumed. Episode IDs retain at least six digits and expand naturally after
`episode_999999`.

Inspect command options or validate an existing completed collection:

```bash
./uav expert-collect --help
./uav expert-validate --episodes 100
```

`--dry-run` is an offline developer fixture. It exercises lifecycle, progress,
manifest, and seed logic below `run_logs/` without starting Isaac, ROS, PX4, or
creating the formal dataset.

## Frozen research contract

Every formal episode uses normal mode from
`isaac/runtime/episode_scene.py`: start `(0, 0)`, goal `(3, 5)`, exactly eight
high-rise buildings, the frozen width/depth/height/yaw ranges, 0.50 m minimum
gap, two guaranteed direct-path blockers, and canonical lighting. The collector
validates the pure deterministic scene before starting the flight and the
aggregate validator checks the recorded scene again.

Camera geometry is not configured by the collector. It starts the existing
canonical sensor runtime with:

- FPV RGB: 320×180 JPEG, quality 85, approximately 5 Hz primary stream.
- FPV Depth: raw uint16-millimetre PNG, approximately 5 Hz auxiliary stream.
- Observer RGB: existing Observer geometry, approximately 2 Hz auxiliary
  stream.

The BC V1 sample remains unchanged: current FPV preprocessing and frozen
encoder produce 64 latent values; body velocity (2), body-frame goal direction
(2), normalized distance (1), and previous normalized action (3) produce the
72D observation. The target remains normalized
`[v_forward, v_right, yaw_rate]` with the existing action limits.

## Automated lifecycle and failures

For each planned seed the tool validates the scene, starts an isolated
Isaac/PX4/XRCE runtime, proves landed/disarmed readiness, applies the scene,
runs the finite guarded ASTAR_EXPERT flight, records synchronized streams,
lands, attaches safe terminal evidence, validates the episode, cleans all owned
process groups, and advances automatically.

Normal mission failures (collision/tracking, blocked scene, safe A*/goal
failure, image QA failure, or episode dataset validation failure) are finalized,
recorded as rejected attempts, and do not stop later seeds. Their episode
directories remain in place and `rejected_attempts/attempt_XXXXXX.json` indexes
the seed, category, reason, episode/flight/validation evidence, and runtime log.
Missing recorder/evidence, unsafe terminal state, corrupt filesystem output,
runtime readiness failure, internal exception, or loss of process ownership is
an infrastructure failure; the batch aborts and leaves a resumable manifest.

Planner readiness accepts both a separately validated B-spline and the
planner's collision-checked `ASTAR_FALLBACK` final path. A rejected B-spline
therefore does not discard a valid A* route. The formal takeoff supervisor's
0.25 m altitude tolerance also matches the trajectory follower's 0.25 m
terminal-position tolerance, preventing a settled takeoff trajectory from
remaining just below the mission-transition boundary.

`collection_manifest.json` is the resume source of truth and complete audit
trail for accepted and rejected attempts. `dataset_manifest.json` is the
BC-facing accepted manifest and contains only accepted episode IDs.
`collection_summary.json` records requested, attempted, accepted, rejected,
rejection categories, infrastructure failures, and completion state.

## Progress and Visual QA

Progress output is driven by a recorder `progress.json` snapshot written at
only 1 Hz and read by the orchestration process. It does not subscribe to or
delay sensor callbacks. The terminal reports episode/total, percentage, seed,
flight state, success/failure counts, current and total accepted samples,
rejections, dataset size, elapsed time, ETA, and concise state transitions.

After every 20 completed episodes, the tool builds a 3×3 contact sheet from a
recent successful episode:

```text
FPV start / mid-flight / near-goal
Observer start / mid-flight / near-goal
Depth start / mid-flight / near-goal
```

Contact sheets and their source-path JSON are stored under
`visual_qa/` inside the dataset. QA generation failure is recorded and does not
interrupt safe flight collection, but final aggregate validation rejects a
collection with missing scheduled sheets.

## Validation coverage

Per-episode and aggregate validation checks unique seeds and scenes, the
canonical eight-building/two-blocker scene, building bounds and gaps, lighting,
JPEG readability and size, uint16 PNG depth, timestamp monotonicity,
synchronization tolerances, rejection accounting, stream counts, A* path
metadata, terminal outcome and safe failure evidence, sampling rate, disk
statistics, and Visual QA cadence. Every accepted sample is decoded and passed
through the current preprocessing/encoder to reconstruct a finite 64D latent,
72D observation, and 3D expert target.

This tool does not start BC or PPO training.
