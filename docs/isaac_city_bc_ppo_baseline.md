# Isaac fixed-height city BC → PPO baseline

This baseline turns the existing fixed-altitude/random-high-rise idea into one
reproducible Gymnasium-style `reset(seed)` / `step(action)` task. Isaac renders
clean FPV RGB, while `FixedHeightCityCore` owns the deterministic kinematic
dynamics, collision checks, termination, reward, and A* expert. No target,
waypoint, path, or debug markers are rendered into the policy image.

This is deliberately a minimum high-level navigation environment. It controls
body velocity and yaw rate at a fixed altitude; it is not yet a rigid-body UAV,
motor controller, GPS-failure detector, inspection task, or return-to-carrier
mission.

## Reproduce in the required order

```bash
./uav isaac-city-smoke --headless --enable_cameras
./uav isaac-bc-collect --headless --enable_cameras
./uav latent-prepare
./uav latent-bc-train
./uav bc-closed-loop --headless --enable_cameras
./train_ppo.sh 50k
```

For the usual longer experiment, `./train_ppo.sh 50k` is the recommended
entry point. `quick`, `50k`, and `100k` presets are available, as is a raw
positive timestep count such as `./train_ppo.sh 25000`. It automatically uses
the BC baseline, clean headless Isaac camera mode, standard hyperparameters,
and a timestamped output directory. Training and final evaluation display
interactive progress bars; pass `--no-progress` when redirecting output to a
plain log file.

Every completed rollout is atomically saved as `latest.pt`. Pressing `Ctrl+C`
also writes an emergency checkpoint, prints its absolute output directory, and
shows the exact resume command. If a later run is interrupted, resume it with
the same target timestep count:

```bash
./train_ppo.sh resume training_runs/ppo_city_50k_YYYYMMDD_HHMMSS 50k
```

After a normal final evaluation, the runner automatically writes
`training_metrics.png` beside `summary.json` and prints both absolute paths.
The chart contains BC train/validation MSE, PPO cumulative training success by
environment timestep, and the held-out BC-versus-final-PPO comparison.

Random cities are accepted only when A* confirms start-to-goal reachability.
If the first layout for a seed is blocked, the deterministic RNG stream
generates the next layout. This keeps PPO and demonstration episodes solvable
without silently changing the external episode seed.

The collector saves the same-step pair in this order: capture observation
`t`, calculate A* action `t`, save the pair, apply action `t`, then advance
Isaac. The formal dataset has 24/6/6 train/validation/test episodes (1,768,
447, and 434 samples); all 36 A* episodes succeeded and were marked
synchronized. Latent mean/std are calculated from the training split only.

## Contracts

RGB is `uint8 [72,128,3]`. The frozen Autoencoder maps normalized channel-first
RGB `[B,3,72,128]` to latent `[B,64]`. The actor input is latent 64 plus state 8:
body velocity 2, body-frame goal unit direction 2, normalized goal distance 1,
and previous normalized action 3. Train-only mean/std standardize all 72 values.

Both BC and PPO output normalized `[forward_velocity, right_velocity, yaw_rate]`
in `[-1,1]`. Physical maxima are 1.0 m/s, 0.8 m/s, and 1.0 rad/s. Keeping the
same 72→3 actor makes PPO initialization an exact state-dict copy rather than
an approximate layer conversion.

BC minimizes equal-component action MSE. Normalized action ranges make the
three terms comparable, and MSE is the direct Gaussian/expert-regression
baseline. PPO minimizes the negative clipped surrogate plus `0.5 * value MSE`
minus `0.001 * entropy`, with clip 0.2. Clipping limits destructive policy
updates from the useful BC start; the critic predicts discounted return; the
small entropy term retains controlled exploration. The frozen encoder is not
updated by either method.

The per-step environment reward is:

```text
3.0 * (previous_goal_distance - new_goal_distance)
- 0.01
- 0.02 * sum((action - previous_action)^2)
+ 20 on success
- 20 on collision or out-of-bounds
- 2 on timeout
```

Progress gives dense directional feedback; the step cost discourages loitering;
smoothness discourages rapid control changes; the symmetric terminal rewards
make reaching the goal dominate and collision clearly undesirable.

## Recorded baseline result

BC open-loop test MSE is 0.01777. Closed-loop BC succeeded on 13/20 seeds
starting at 30000. On the comparison seeds starting at 50000, BC reached 9/20.
The BC-initialized PPO runs produced:

| Training steps | Evaluation episodes | Success | Collision | Timeout |
|---:|---:|---:|---:|---:|
| 8,192 | 20 | 40% | 60% | 0% |
| 50,000 | 50 | 52% | 44% | 4% |
| 100,000 | 100 | 54% | 43% | 3% |

The longer runs improve on the first 8,192-step PPO result, but they do not
establish a statistically controlled improvement over BC because BC has not
yet been evaluated on the same 50- or 100-episode set. These artifacts prove a
reproducible BC-initialized PPO pipeline, not robust navigation performance.
The next experiment should use checkpoint gating and more diverse
on-policy/DAgger data rather than treating additional timesteps alone as the
solution.

The Autoencoder checkpoint was trained on the earlier RGB collection, whereas
the current demonstrations use marker-free Isaac images. That domain mismatch
and the small single-environment rollout budget are the main limitations of
this first end-to-end baseline.

The source-controlled checksum and local-artifact audit is recorded in
[`ml_artifact_inventory.md`](ml_artifact_inventory.md). Generated datasets and
model checkpoints remain intentionally excluded from Git.
