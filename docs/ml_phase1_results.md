# ML Phase 1 results

## Classification

`SYNTHETIC_SOFTWARE_PIPELINE_VALIDATED_REAL_ISAAC_BRIDGE_PENDING`

The mandatory software milestone is implemented. No real Isaac expert dataset
and no closed-loop simulator navigation rollout are claimed.

## Baseline and audit

- Requested starting HEAD: `036791da3852530b8e6283ef8c9dbe6025089ea8`.
- Branch: `feature/ml-training-pipeline`.
- The engineering branch was observed at `6ccf97a` but was not rewritten; the
  new branch was created directly from the requested stable commit.
- Reused components: pure A*, safe B-spline/A* fallback, timed trajectory,
  sampler, bounded trajectory follower, coordinate conversions, and offline
  kinematic fixture.
- Legacy Isaac scene/camera/recorder/episode/A* files were inspected and left
  untouched.
- NavRL's environment, observation/action spec, LiDAR path, PPO, and vectorized
  collector were inspected as references only.

## Verified software evidence

The canonical smoke command is:

```bash
./uav bc-smoke-test --device cuda --epochs 30
```

The first recorded run on the local RTX 4080 SUPER used 4 synthetic episodes
and 40 samples:

| Measure | Value |
|---|---:|
| initial train action MSE | 0.569080 |
| final train action MSE | 0.129488 |
| train loss ratio | 0.227538 |
| initial validation action MSE | 0.684124 |
| final validation action MSE | 0.568545 |
| parameters | 22,876 |

Forward/backward, validation, CSV history, full checkpoint save/reload, same
sample inference, deterministic eval mode, and output clipping passed. Smoke
artifacts use a temporary directory and are removed; generated persistent
datasets/checkpoints are intentionally ignored by Git.

These values demonstrate pipeline mechanics and tiny-set overfitting only.
They are not UAV learning quality or navigation success metrics.

An additional persistent, Git-ignored pipeline run used 8 synthetic episodes,
192 synchronized samples (6/144 train, 2/48 validation), and 20 epochs:

| Measure | Value |
|---|---:|
| initial train action MSE | 0.110030 |
| final train action MSE | 0.050931 |
| initial validation action MSE | 0.111868 |
| final validation action MSE | 0.079746 |
| depth min / max / mean (m) | 0.590010 / 20.000000 / 18.444139 |
| near-obstacle sample fraction | 0.057292 |

The reload-verified artifact paths are
`checkpoints/bc_v0_synthetic.pt` and
`training_runs/bc_v0_synthetic_history.csv`. They remain local generated
artifacts and are excluded from source control.

## Real dataset and rollout status

Real episodes/samples: `0 / 0`. The exact missing item is a simulator-direct
Isaac 5.1/Isaac Lab adapter that captures metric 64x64 depth and state in one
simulation step, invokes the pure expert on that same snapshot, saves the
synchronized record, then applies the velocity command to a simulation-only
controller. The existing RGB PNG/wall-time logger is insufficient and the old
PX4 runner is outside scope.

Closed-loop BC rollout: not implemented. PPO: not implemented. ROS policy node:
not implemented. PX4 output added: no.
