# FPV/TOP/depth Autoencoder baseline

## Purpose and boundary

This is Stage A of the two-stage baseline: train `RgbAutoencoderV0` with
reconstruction MSE. Stage B freezes its encoder and trains BC separately.
There is no joint reconstruction/action loss or navigation reward here.

## Data split

The default dataset is the formal expert collection. `fpv_rgb`, `top`, and
`fpv_depth` all use the complete formal TOP comparison cohort. Splits are
deterministic, disjoint at episode level, and use the same 80/10/10 algorithm
and seed as BC. The older legacy dataset remains available by explicitly
passing `--dataset ./uav_vision_dataset --split-file
uav_vision_dataset/_audit/autoencoder_split.json` with `fpv_rgb`.

| Split | Episodes | FPV frames |
|---|---:|---:|
| Train | 19 | 3,209 |
| Validation | 4 | 722 |
| Test | 4 | 730 |

Each split covers baseline, natural, forced, and city environments. Validation
selects the checkpoint; test is evaluated once after training.

## Input, latent, and output

RGB sources use RGB conversion, bilinear 128x72 resize, CHW layout, and
`[0,1]` scaling. Depth uses recorded uint16 millimetres, maps invalid zero to
zero, clips valid values to 50--30000 mm, scales to `[0,1]`, and repeats the
single channel three times to retain the existing architecture and 64D latent.

```text
model input:    [batch, 3, 72, 128] float32 RGB in [0,1]
encoder output: [batch, 64] float32 latent vector (unbounded)
decoder output: [batch, 3, 72, 128] float32 RGB reconstruction in [0,1]
loss:           mean squared error over every RGB pixel/channel
```

The future PPO actor should receive the 64D encoder output, not the reconstructed
image. It must concatenate latent data with local velocity, relative goal state,
and mission phase. The decoder is only required for representation pretraining
and visual diagnostics.

## Architecture

Four stride-2 convolutions reduce the image to 128x5x8. A linear layer creates
the 64D latent. A symmetric linear/transposed-convolution decoder reconstructs
the image. The model has 891,811 trainable parameters.

## Recorded baseline

Command:

```bash
./uav ae-train \
  --dataset ./uav_vision_dataset \
  --split-file uav_vision_dataset/_audit/autoencoder_split.json \
  --image-source fpv_rgb \
  --epochs 20 \
  --batch-size 128 \
  --workers 4 \
  --device cuda \
  --output-dir autoencoder_runs/rgb_ae_v0_baseline_20260811
```

Use a source-matched encoder for Stage B:

```bash
./uav ae-train --dataset bc_expert_cube --epochs 200
./uav bc-train --dataset bc_expert_cube --epochs 500
```

TOP is the default image source. AE output is automatically placed under
`artifacts/experiments/autoencoder/<dataset>/top/run_<timestamp>`, and a
completed provenance index lets BC select the matching encoder without a glob.
The advanced `--image-source`, `--output-dir`, and BC `--encoder` overrides
remain available.

Each training CLI starts a managed localhost TensorBoard server by default and
keeps it running after successful training until Ctrl+C; use `--no-tensorboard`
for tests or batch jobs. Each run writes
`reconstruction_loss_curves.png`, preserves the legacy
`loss_curve.png`, and writes TensorBoard scalars
`ae/train_reconstruction_loss` and
`ae/validation_reconstruction_loss`. Fixed validation samples are logged as
`ae/<image_source>/validation_original_vs_reconstructed` at the configured
image interval. Start TensorBoard with:

After training, the retained events can still be reopened manually with
`tensorboard --logdir <run-directory>/tensorboard`.

Best checkpoint was epoch 20:

| Split | MSE | MAE | PSNR (dB) |
|---|---:|---:|---:|
| Train | 0.004703 | 0.043084 | 23.276 |
| Validation | 0.005044 | 0.043823 | 22.972 |
| Test | 0.005620 | 0.046205 | 22.502 |

The test-minus-train MSE gap is 0.000918. There is a modest generalization gap,
but no sustained validation-loss divergence in 20 epochs. Epoch 18 had a
temporary optimization spike before recovery.

Reconstructions preserve large scene layout and prominent markers, but thin
obstacles and sharp boundaries are blurry. Pixel MSE is dominated by sky and
ground area and does not prove the latent is collision-relevant. This model is
therefore a reproducible baseline for comparison, not yet the final PPO visual
encoder.
