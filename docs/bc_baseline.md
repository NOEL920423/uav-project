# BC baseline training and evaluation

This is the formal first behavior-cloning reference for the canonical
high-rise expert dataset. It deliberately remains small and reproducible:

```text
FPV RGB / TOP RGB / FPV depth
  -> source-specific preprocessing to 3x72x128 float32 [0,1]
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
right and 1.0 rad/s yaw. Only the selected image stream is used. The policy
receives no map, obstacle truth, full scene state, reward, or auxiliary target.

`LatentBcPolicy` is reused because it already implements the required
72D-to-3D MLP. The older class named `BcPolicyV0` has a depth/state input and
four-axis action contract, so using it here would violate the formal dataset
contract and create an incompatible parallel representation.

## Training

Run training explicitly after the dataset collection has completed and passed
its validator:

```bash
./uav bc-train --dataset bc_expert_cube --epochs 500
./uav bc-train --dataset bc_expert_cube --image-source fpv_rgb --epochs 100 \
  --encoder <fpv-ae-run>/best.pt
./uav bc-train --dataset bc_expert_cube --image-source fpv_depth --epochs 100 \
  --encoder <depth-ae-run>/best.pt
./uav bc-train --help
```

Defaults are the formal dataset at
`artifacts/datasets/bc_expert_cylinder_v1`, the existing frozen encoder at
`autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt`, Adam with learning rate
`1e-3`, batch size 64, at most 100 epochs, patience 12, and a fixed seed. TOP
is the default image source. Without `--encoder`, BC reads only the completed
matching AE provenance index for the same dataset/source and validates its
summary, checkpoint metadata, hash, preprocessing, architecture, and 64D
latent contract. It never falls back to another source or dataset.

FPV RGB is read from `samples.csv.image_path`. TOP RGB and FPV depth are joined
from `auxiliary.csv.observer_rgb_path` and `fpv_depth_path` by exact
`episode_id`/`sample_id` identity and primary timestamp. Availability, matched
status, timestamp error, and files are checked; there is no source fallback.
All three sources use the complete `fixed_global_top` comparison cohort and
therefore receive the same deterministic split for the same seed. Sparse
legacy TOP episodes are explicitly excluded from this comparison cohort.

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
  tensorboard/events.out.tfevents.*
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

BC output is automatically placed under
`artifacts/experiments/bc/<dataset>/<source>/run_<timestamp>`. The CLI starts a
managed TensorBoard server by default and keeps it running after successful
training until Ctrl+C; `--no-tensorboard` disables only the server, not
event-file recording. TensorBoard records
`bc/train_action_loss` and
`bc/validation_action_loss` each epoch. It records
`bc/test_forward_rmse`, `bc/test_right_rmse`, and
`bc/test_yaw_rate_rmse` once after reloading `best.pt`. Start it with:

Retained events can later be reopened with
`tensorboard --logdir <run-directory>/tensorboard`.

For a remote server, use an SSH tunnel such as
`ssh -L 6006:localhost:6006 user@server`, then open
`http://localhost:6006` locally. TOP/depth checkpoints are offline baselines;
the current closed-loop runtime is fail-closed because it supplies FPV RGB.

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
