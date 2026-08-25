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

Add 100 episodes to a new dataset:

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube
```

If that 100-episode run is interrupted, resume it with the same count:

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube --resume
```

`--episodes` is the number of episodes to add in the current run, not a fixed
dataset-wide target. After a run completes, any positive count can be appended
without `--resume`. For example, this adds 5 episodes to an existing dataset:

```bash
./uav expert-collect --episodes 5 --dataset bc_expert_cube
```

If that 5-episode run stops partway through, use
`./uav expert-collect --episodes 5 --dataset bc_expert_cube --resume`. The
resume count must match the
unfinished run so the tool cannot accidentally create a second plan or skip
episodes. When no run is unfinished, `--resume` is also accepted and starts a
new append run. There is no configured total-episode ceiling; IDs retain at
least six digits and expand naturally after `episode_999999`.

The manifest's accumulated `target_episodes` grows after each append. Completed
episode directories are never overwritten. Any incomplete directory is moved
to the Git-ignored recovery log before that episode is retried with its
original seed.

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

Normal mission failures (for example a safe A* or goal failure) are finalized,
validated for landed/disarmed evidence, recorded as failed episodes, and do not
stop later episodes. Missing recorder/evidence, unsafe terminal state, runtime
readiness failure, or loss of process ownership is an infrastructure failure;
the batch stops and leaves a resumable manifest.

Planner readiness accepts both a separately validated B-spline and the
planner's collision-checked `ASTAR_FALLBACK` final path. A rejected B-spline
therefore does not discard a valid A* route. The formal takeoff supervisor's
0.25 m altitude tolerance also matches the trajectory follower's 0.25 m
terminal-position tolerance, preventing a settled takeoff trajectory from
remaining just below the mission-transition boundary.

`collection_manifest.json` is the resume source of truth. It stores the
append-only episode/seed plan, collection-run and append history, per-episode
lifecycle status, accepted/rejected counts,
terminal reason, outcome, dataset bytes, Visual QA status, and final aggregate
validation. `dataset_manifest.json` retains the unchanged BC dataset contract.

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
