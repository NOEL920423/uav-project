# BC baseline training and evaluation

This is the formal first behavior-cloning reference for the canonical
high-rise expert dataset. It deliberately remains small and reproducible:

```text
FPV RGB 320x180
  -> PIL RGB, bilinear resize to 128x72, float32 [0,1]
  -> frozen RgbAutoencoderV0
  -> latent64

latent64 + body-state8
  -> train-split mean/std normalization
  -> Linear(72,128), Tanh, Linear(128,128), Tanh,
     Linear(128,3), Tanh
  -> normalized [v_forward, v_right, yaw_rate]
```

The 8D state is body forward/right velocity, body-frame unit goal direction,
goal distance divided by 10 m and clipped to 1, and the previous normalized
three-axis action. Physical action limits remain 1.0 m/s forward, 0.8 m/s
right and 1.0 rad/s yaw. The policy receives no depth, observer camera, map,
obstacle truth, full scene state, or A* action.

`LatentBcPolicy` is reused because it already implements the required
72D-to-3D MLP. The older class named `BcPolicyV0` has a depth/state input and
four-axis action contract, so using it here would violate the formal dataset
contract and create an incompatible parallel representation.

## Training

Run training explicitly after the dataset collection has completed and passed
its validator:

```bash
./uav bc-train
./uav bc-train --epochs 100
./uav bc-train --help
```

Defaults are the formal dataset at
`artifacts/datasets/bc_expert_highrise_v1`, the existing frozen encoder at
`autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt`, Adam with learning rate
`1e-3`, batch size 64, at most 100 epochs, patience 12, and a fixed seed.

Before tensor construction, the tool independently checks the collection and
validation artifacts, successful terminal status, per-episode validation,
encoder hash, sample counts, finite state/action values, normalized action
bounds, synchronization tolerance, and every FPV image. Failed, missing, or
corrupt episodes are excluded; data are never replaced by zeros or synthetic
actions. The terminal prints total, usable and excluded episodes, reasons, and
accepted samples.

The split is deterministic and episode-level, approximately 80/10/10. Frames
from one episode cannot cross train, validation, or test. The test split is
loaded only after the best-validation checkpoint has been selected. The exact
assignment and excluded records are saved in `split_manifest.json`.

Each run is written below:

```text
artifacts/experiments/bc_baseline/run_<UTC timestamp>/
  best.pt
  last.pt
  dataset_audit.json
  split_manifest.json
  training_config.json
  training_history.csv
  metrics.json
  summary.json
  plots/loss_curves.png
  plots/per_action_rmse.png
  plots/expert_vs_predicted.png
```

`best.pt` and `last.pt` include the model and optimizer states, normalization,
configuration, random seed, dataset manifest reference/hash, complete split,
encoder path/hash and observation/action contracts. The encoder is in eval
mode with gradients disabled throughout policy training. A successful run also
updates the ignored
`artifacts/experiments/bc_baseline/latest.json`, which evaluation uses by
default.

Offline metrics include equal-component normalized action MSE plus per-action
MSE, MAE, and RMSE. They answer: **can BC imitate held-out expert actions?**
They do not establish that the UAV can navigate successfully.

## Closed-loop evaluation

After training has produced `best.pt`, run:

```bash
./uav bc-eval --episodes 20
./uav bc-eval --help
```

An explicit checkpoint can be selected with `--checkpoint PATH`. The tool
loads the exact encoder recorded in the checkpoint, verifies its SHA-256,
reuses the training preprocessing and normalization, and launches the existing
headless Isaac fixed-height city environment with cameras enabled. Evaluation
seeds are deterministic and checked against every seed in the expert dataset;
they cannot overlap train, validation, or test expert episodes.

During every rollout, `CONTROL SOURCE = BC_POLICY`. The evaluator calls only
the BC policy for actions; it never calls the environment's A* expert, blends
an expert action, or adds privileged information to the policy observation.
A* remains internal only for generating reachable randomized environments and
is not a control source. Collision and out-of-bounds termination are reported
as collision failures.

Evaluation output is written below:

```text
artifacts/experiments/bc_evaluation/run_<UTC timestamp>/
  metrics.json
  episodes/episode_<index>.json
  plots/closed_loop_outcomes.png
  plots/goal_distance_by_episode.png
```

Each episode records seed, success, collision, timeout, terminal reason,
minimum/final goal distance, flight duration, measured path length, policy
ownership, safety-abort state, and blending state. Aggregate metrics contain
counts/rates and mean distance, duration, and path length. These closed-loop
metrics answer: **can BC actually fly to the goal by itself?** A low offline
MSE must never be interpreted as navigation success.

## Scope and current risk

This baseline does not perform encoder fine-tuning, RGB-D fusion, recurrent or
attention modeling, DAgger, GAIL, PPO, or hyperparameter search. The closed-loop
environment uses deterministic fixed-height dynamics and an Isaac-rendered
camera; it is useful policy-only evidence but does not replace a separately
authorized PX4/Pegasus learned-policy flight or hardware validation. Domain
shift between the formal Pegasus dataset and this evaluation renderer remains
an explicit risk.

Generated datasets, checkpoints, metrics, and plots are ignored by Git.
