# FPV RGB Autoencoder baseline

## Purpose and boundary

This baseline tests whether the legacy FPV RGB collection can train a compact
image representation. It does not train a flight policy and does not report
navigation success. The legacy images expose simulator-only red goal and cyan
path/waypoint geometry, so this result is not map-free navigation evidence.

## Data split

The audit-selected data uses A* expert FPV only. TOP observer images and two BC
policy rollouts are excluded. Splits are disjoint at the episode level:

| Split | Episodes | FPV frames |
|---|---:|---:|
| Train | 19 | 3,209 |
| Validation | 4 | 722 |
| Test | 4 | 730 |

Each split covers baseline, natural, forced, and city environments. Validation
selects the checkpoint; test is evaluated once after training.

## Input, latent, and output

External input is one FPV RGB frame. The loader converts it to RGB, resizes the
legacy 960x540 frame to 128x72 with bilinear interpolation, converts HWC to CHW,
and scales bytes to `[0,1]` float32.

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
  --epochs 20 \
  --batch-size 128 \
  --workers 4 \
  --device cuda \
  --output-dir autoencoder_runs/rgb_ae_v0_baseline_20260811
```

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
