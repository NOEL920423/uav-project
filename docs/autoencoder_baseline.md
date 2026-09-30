# FPV/TOP/depth Autoencoder 基準流程

## 目的與範圍

這是兩階段基準流程的 Stage A：使用 reconstruction MSE 訓練 `RgbAutoencoderV0`。
Stage B 會固定 encoder，另外訓練 BC。此流程不會共同最佳化 reconstruction/action loss，
也不使用 navigation reward。

## 資料切分

預設 dataset 是正式 expert collection。`fpv_rgb`、`top` 和 `fpv_depth` 都使用完整的
正式 TOP comparison cohort。資料依 episode 切分，結果具確定性且各 split 不重疊，並與 BC
共用相同的 80/10/10 演算法和 seed。舊版 legacy dataset 仍可透過 `fpv_rgb` 明確指定：
`--dataset ./uav_vision_dataset --split-file uav_vision_dataset/_audit/autoencoder_split.json`。

| Split | Episodes 數 | FPV frames 數 |
|---|---:|---:|
| Train | 19 | 3,209 |
| Validation | 4 | 722 |
| Test | 4 | 730 |

每個 split 都包含 baseline、natural、forced 和 city environments。Validation split 用於
選擇 checkpoint；test split 只在訓練完成後評估一次。

## Input、latent 與 output

RGB sources 會轉為 RGB、以 bilinear 方式 resize 到 128x72、使用 CHW layout，並縮放至
`[0,1]`。Depth 使用記錄的 uint16 millimetres；無效的零值維持為零，有效值裁切至
50--30000 mm 後縮放到 `[0,1]`，再將單一 channel 複製三次，以沿用既有架構和 64D latent。

```text
model input:    [batch, 3, 72, 128] float32 RGB in [0,1]
encoder output: [batch, 64] float32 latent vector (unbounded)
decoder output: [batch, 3, 72, 128] float32 RGB reconstruction in [0,1]
loss:           mean squared error over every RGB pixel/channel
```

未來的 PPO actor 應接收 64D encoder output，而不是重建影像。它必須將 latent 與 local
velocity、relative goal state 及 mission phase 串接。Decoder 只用於 representation
pretraining 和視覺診斷。

## 架構

四個 stride-2 convolutions 將影像縮小至 128x5x8。Linear layer 產生 64D latent；對稱的
linear/transposed-convolution decoder 負責重建影像。模型共有 891,811 個可訓練參數。

## 已記錄的基準結果

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

Stage B 請使用影像來源相符的 encoder：

```bash
./uav ae-train --dataset bc_expert_cube --epochs 200
./uav bc-train --dataset bc_expert_cube --epochs 500
```

TOP 是預設 image source。AE output 會自動寫入
`artifacts/experiments/autoencoder/<dataset>/top/run_<timestamp>`；完成的 provenance
index 可讓 BC 不透過 glob 就選到相符 encoder。仍可使用進階選項 `--image-source`、
`--output-dir` 與 BC 的 `--encoder` 覆寫預設值。

每個訓練 CLI 預設會啟動受管理的 localhost TensorBoard server，並在成功訓練後持續執行，
直到按下 Ctrl+C。測試或 batch jobs 可使用 `--no-tensorboard`。每次 run 只會產生
`reconstruction_loss_curves.png` 作為 loss plot。Legacy summary artifact key `loss_curve`
也指向同一個檔案。每次 run 會寫入以下 TensorBoard scalars：
`ae/train_reconstruction_loss` 和 `ae/validation_reconstruction_loss`。固定的 validation
samples 會依設定的 image interval 記錄為
`ae/<image_source>/validation_original_vs_reconstructed`。啟動 TensorBoard 可使用：

訓練後仍可使用以下命令手動重新開啟保留的 events：
`tensorboard --logdir <run-directory>/tensorboard`.

最佳 checkpoint 來自 epoch 20：

| Split | MSE | MAE | PSNR (dB) |
|---|---:|---:|---:|
| Train | 0.004703 | 0.043084 | 23.276 |
| Validation | 0.005044 | 0.043823 | 22.972 |
| Test | 0.005620 | 0.046205 | 22.502 |

Test 與 train 的 MSE 差距為 0.000918。存在輕微的 generalization gap，但 20 個 epochs
內沒有持續的 validation-loss 發散。Epoch 18 曾短暫出現 optimization spike，之後恢復。

重建結果保留主要場景布局和明顯標記，但細小障礙物與銳利邊界較模糊。Pixel MSE 主要
受天空和地面區域影響，不能證明 latent 能辨識碰撞風險。因此此模型是可重現的比較基準，
尚不是 PPO 最終使用的 visual encoder。
