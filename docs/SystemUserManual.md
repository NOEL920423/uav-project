# UAV 系統使用手冊

本手冊說明從專家資料收集、資料驗證、Autoencoder 訓練、BC 訓練，到 Isaac Sim / Pegasus / PX4 SITL closed-loop 飛行評估的完整流程。所有指令都在 repository 根目錄執行。

## 流程摘要

```text
expert-collect
  -> expert-validate
  -> ae-train
  -> bc-train
  -> bc-eval
```

資料收集和 closed-loop 評估會啟動模擬與飛行流程；AE 和 BC 訓練及資料驗證是離線工作。開始前請確認專案所需的 Isaac Sim、Pegasus、PX4 SITL、ROS 2 和 Python 環境已安裝並可由 `./uav` 使用。

## 0. 設定本次資料集名稱

```bash
# 設定本次資料集名稱，以下流程會沿用這個變數。
DATASET=bc_expert_run_v1
```

這是 shell 變數，以下命令會將資料寫入 `artifacts/datasets/bc_expert_run_v1`。新的一輪收集請使用新的資料集名稱，避免將不同實驗混在一起。

**命令來源：**`./uav` 的 expert collection CLI；參數說明見 [`expert_dataset_collection.md`](expert_dataset_collection.md) 與 `uav_ml/tools/expert_collect.py`。

## 1. 收集專家資料

```bash
# 收集 1000 筆成功且通過單筆檢查的 episodes；最多嘗試 1500 次。
./uav expert-collect --episodes 1000 --max-attempts 1500 --dataset "$DATASET"
```

`--episodes` 是資料集累計的 accepted episode 目標，不是嘗試次數。若中斷後要接續同一份資料集，使用：

```bash
# 接續收集，直到同一資料集累計達到 1000 筆 accepted episodes。
./uav expert-collect --episodes 1000 --max-attempts 1500 --dataset "$DATASET" --resume
```

可先查詢 collector 支援的選項：

```bash
# 顯示 expert collector 的完整參數說明。
./uav expert-collect --help
```

**命令來源：**[`expert_dataset_collection.md`](expert_dataset_collection.md)、`uav` 的 `expert-collect` dispatch，以及 `uav_ml/tools/expert_collect.py`。collector 會管理 Isaac / PX4 episode 流程、記錄影像與飛行證據，並在收集結束時執行 aggregate validation。

## 2. 驗證並完成資料集

若 collector 已正常結束並將資料集標為 complete，可直接進入下一階段。若收集達到目標後因中斷或 aggregate validation 錯誤而未完成，先停止該資料集的 collector，再執行離線驗證與 finalize：

```bash
# 重新驗證 1000 筆 accepted episodes；通過後將 collection metadata 標記為 complete。
./uav expert-validate \
  --dataset "artifacts/datasets/$DATASET" \
  --episodes 1000 \
  --autoencoder autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt \
  --finalize
```

此驗證需要一個既有 AE checkpoint，因為 validator 會將影像送入 encoder，確認可以重建有限的 64D latent 與 72D BC observation；它不會用這個 checkpoint 訓練新 AE。若預設 checkpoint 不存在，請將 `--autoencoder` 改成一個相容的既有 AE checkpoint。驗證有錯誤時不要開始訓練，先依錯誤訊息修復或檢查資料。

**命令來源：**`uav` 的 `expert-validate` dispatch、`uav_ml/tools/validate_expert_collection.py` 和 `uav_ml/tools/validate_expert_episode.py`。一般選項可用 `./uav expert-validate --help` 查詢。

## 3. 訓練 Autoencoder

```bash
# 使用 TOP RGB 訓練 200 epochs 的 Autoencoder；TOP 也是預設影像來源。
./uav ae-train --dataset "$DATASET" --image-source top --epochs 200
```

AE 會使用 episode-level train / validation / test split，並將訓練結果寫入 `artifacts/experiments/autoencoder/` 下的 run 目錄。記下輸出的 run 路徑；BC 階段會依 dataset 和 image source 選用相符的已完成 AE checkpoint。需要調整 batch size、GPU 或輸出位置時，可查詢：

```bash
# 查看 Autoencoder 訓練支援的參數。
./uav ae-train --help
```

**命令來源：**[`autoencoder_baseline.md`](autoencoder_baseline.md)、`uav` 的 `ae-train` dispatch，以及 `uav_ml/train_autoencoder.py`。

## 4. 訓練 Behavior Cloning 模型

```bash
# 設定本次 BC run 輸出目錄；請確認它尚不存在，且每次訓練使用新名稱。
BC_RUN="artifacts/experiments/bc_baseline/${DATASET}_run_01"
# BC 會依 dataset 與 image source 自動選取已完成且相符的 AE provenance。
# 使用 TOP 影像和同資料集的 AE encoder 訓練 BC 500 epochs。
./uav bc-train --dataset "$DATASET" --image-source top --epochs 500 --output "$BC_RUN"
```

BC 預設輸出在 `artifacts/experiments/bc_baseline/`。完成後記下這次 BC run 的 `best.pt` 路徑，供 closed-loop 評估使用。可查詢其餘訓練參數：

```bash
# 查看 BC 訓練參數與 encoder 選項。
./uav bc-train --help
```

**命令來源：**[`bc_baseline.md`](bc_baseline.md)、`uav` 的 `bc-train` dispatch，以及 `uav_ml/tools/bc_baseline.py`。

## 5. Closed-loop 飛行評估

```bash
# 使用訓練完成的 BC checkpoint 執行 20 次 TOP RGB closed-loop 飛行。
./uav bc-eval \
  --image-source top_rgb \
  --episodes 20 \
  --checkpoint "$BC_RUN/best.pt"
```

`bc-eval` 會管理 Isaac Sim、Pegasus、PX4 SITL 與 ROS 2 flight stack；若要查看模擬畫面，可加上 `--visible`。評估結果會寫入 `artifacts/evaluations/bc_flight/run_<timestamp>/`。檢查每回合的 success、failure reason 與 aggregate success rate；offline BC loss 不能取代 closed-loop 飛行結果。

```bash
# 查看 closed-loop evaluator 的完整參數。
./uav bc-eval --help
```

**命令來源：**[`bc_baseline.md`](bc_baseline.md)、`docs/architecture.md`、`uav` 的 `bc-eval` dispatch，以及 `uav_ml/tools/bc_flight_evaluation.py`。

## 6. 結果位置

| 階段 | 主要輸出位置 |
|---|---|
| 專家資料 | `artifacts/datasets/<dataset>/` |
| AE checkpoint 和訓練紀錄 | `artifacts/experiments/autoencoder/<dataset>/<image_source>/run_<timestamp>/` |
| BC checkpoint、split 和 metrics | 本手冊指令設定的 `$BC_RUN/` |
| Closed-loop episodes、metrics 和 plots | `artifacts/evaluations/bc_flight/run_<timestamp>/` |

## 執行注意事項

- Collection 和 `bc-eval` 會啟動模擬飛行流程；請先確認 Isaac Sim、PX4 SITL 和 ROS 2 runtime 可用。
- `expert-validate --finalize` 會掃描並驗證整份 collection，可能需要較長時間；它不會收集新 episode。
- 如果資料集已是 `complete` 且有有效的 aggregate validation，不需要再次執行 `--finalize`。
- 新的收集實驗請使用新的 dataset 名稱；resume 時必須沿用同一名稱。
- 每次 BC 訓練請確認使用的 AE encoder 影像來源與訓練設定相符。
