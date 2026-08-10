# BC v0 dataset contract

Dataset version is `bc_v0.1`. Generated data lives under `datasets/` and is
ignored by Git.

```text
datasets/bc_v0/
  metadata.json
  train/episode_<id>.npz
  validation/episode_<id>.npz
```

An episode container avoids millions of small files and requires only NumPy.
Pickle is disabled on load. Each `.npz` contains:

| Array | Shape |
|---|---:|
| `depth` | `T x 1 x 64 x 64` |
| `velocity` | `T x 3` |
| `goal_direction` | `T x 3` |
| `expert_action` | `T x 4` |
| `step` | `T` contiguous from zero |
| `timestamp_s` | `T` strictly increasing |
| `goal_distance_m` | `T` |

Episode metadata in the same container includes scalar `episode_id`,
`scene_seed`, and `planner_path_source`, plus `start_ned[3]`, `goal_ned[3]`,
and `obstacles_north_east_radius[N,3]`. Root metadata records all contract
versions, fixture/real classification, frames, split seed, storage guards, and
the synchronization rule.

## Synchronization rule

At logical simulator step `t`:

1. Freeze/read one state snapshot and its depth output.
2. Sample the timed expert reference for `t`.
3. Compute the bounded follower action from that same state/reference.
4. Append all observation, action, step, and timestamp arrays together.
5. Only then apply the action and advance to `t+1`.

A sample is rejected if its arrays do not have identical `T`. Independent
camera and command processes joined later by nearest wall time are not valid
for this dataset version.

## Split and determinism

The generator deterministically hashes `split_seed:episode_id`, assigns whole
episodes to approximately 80% train and 20% validation, and guarantees at least
one episode in each split. Frame-level random splitting is prohibited. File
discovery and dataset indexing use lexical order, so non-shuffled loading is
deterministic.

## Validation and storage guards

The validator enforces finite values, exact shapes, float32 depth, depth range,
4D actions, `px4_ned` action frame, required metadata, synchronized counts,
increasing timestamps, and zero split overlap. It reports episodes, samples,
action mean/std, velocity ranges, depth min/max/mean, near-obstacle fraction,
and goal-distance distribution.

Generators require explicit episode and maximum-step limits; the fixture also
records sample stride and estimated uncompressed depth bytes. A real generator
must provide equivalent guards before long runs.

