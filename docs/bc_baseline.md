# BC 基準模型訓練與評估

這是使用 canonical 高樓專家資料集建立的第一個正式 behavior-cloning 基準。流程刻意
保持精簡且可重現：

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

8D state 包含 body forward/right velocity、body-frame unit goal direction、除以 10 m
並裁切至 1 的 goal distance，以及前一個正規化三軸 action。實際 action 上限維持為
forward 1.0 m/s、right 0.8 m/s、yaw 1.0 rad/s。Policy 只使用所選影像資料流，不會
接收 map、障礙物真值、完整場景狀態、reward 或 auxiliary target。

沿用 `LatentBcPolicy`，因為它已實作所需的 72D-to-3D MLP。舊類別 `BcPolicyV0` 使用
depth/state input 和四軸 action contract，若用於此處會違反正式資料集 contract，並建立
不相容的另一套表示法。

## 訓練

### 選用的 frozen ResNet18

預設仍使用 AE；既有 `ae-train`、`bc-train --encoder <path>` 和 AE checkpoints 均維持可用。
使用 `--encoder-type resnet18` 選擇 ResNet18。此模式使用 ImageNet `IMAGENET1K_V1` 權重，
固定所有 encoder parameters 和 BatchNorm statistics，只訓練現有 BC MLP。不需要 AE
pretraining 或 reconstruction loss。

Encoder 接收範圍為 `[0,1]` 的 RGB `[B,3,72,128]`，內部套用 ImageNet mean/std，並輸出
`[B,512]`。與原始 state8 串接後得到 `[B,520]`；BC 仍輸出相同的三個正規化 actions。
完整 128x72 影像會保留，不做 center cropping。這是針對 navigation 的解析度選擇，與
torchvision 預設的 224x224 classification crop 不同。請參閱
[ResNet18 官方權重文件](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。

請在 `./uav` 使用的 Python environment 安裝 torchvision。目前專案環境
(`torch==2.12.0`) 對應的版本為：

```bash
python3 -m pip install --no-deps torchvision==0.27.0
```

若 PyTorch 版本不同，請改用相容的 torchvision 版本。AE 不需要 torchvision。第一次
ResNet run 會將 pretrained weights 下載到一般 Torch cache；inference 會載入已保存的
experiment weights，不會再次下載。

先訓練，再執行一次 closed-loop 測試（依需要替換 dataset 名稱）：

```bash
./uav bc-train --dataset bc_expert_cube --image-source top \
  --encoder-type resnet18 --epochs 100 --batch-size 64 --learning-rate 0.001 --no-tensorboard \
  --output artifacts/experiments/bc/bc_expert_cube/top/resnet18/run_01

./uav bc-eval --image-source top_rgb --episodes 1 --visible \
  --checkpoint artifacts/experiments/bc/bc_expert_cube/top/resnet18/run_01/best.pt
```

基本 BC 參數為 `--epochs`、`--batch-size` 和 `--learning-rate`。使用
`--no-tensorboard` 會在訓練後結束，但仍會保存 metrics 和 plots；省略此參數則會讓受管理的
TensorBoard server 持續執行，直到按下 Ctrl+C。`--encode-batch-size` 可獨立控制 encoder
記憶體用量（預設 128）。每次 run 請使用新的 output directory；現有 experiments 不會被
覆寫。未指定 `--output` 時，ResNet runs 會自動寫入
`artifacts/experiments/bc/<dataset>/<image_source>/resnet18/run_<timestamp>`；預設 AE
latest pointer 不變。請將 ResNet 的 `best.pt` 明確傳給 `bc-eval`；它會自動從 checkpoint
讀取 encoder 選擇。請將 `resnet18_encoder.pt` 留在原始 run 目錄：BC checkpoint 和既有 AE
checkpoint 一樣，會記錄其絕對路徑與 SHA-256。

Live evaluator 支援 `top_rgb`、`fpv_rgb` 和 `fpv_depth` checkpoints；每種 checkpoint
只訂閱相符的 Isaac stream，並套用 checkpoint 記錄的 preprocessing。ResNet 訓練可在離線
experiments 使用 `fpv_rgb`，但不支援 `fpv_depth`。使用 `--encoder-type resnet18` 時不可
同時傳入 `--encoder`。

ResNet runs 會產生既有 BC loss curves 和 action-error plots；closed-loop evaluation 會
產生既有 flight plots 和結果 artifacts。Encoder 已固定，因此不會產生 reconstruction plots。

### 使用既有 AE 訓練結果

若 collection 已達 accepted-episode 目標但流程中斷，請先停止 collector，再離線完成驗證
及 metadata 更新：

```bash
./uav expert-validate --dataset artifacts/datasets/bc_expert_cylinder_v2 --finalize
```

此步驟會重新驗證每個 accepted episode，通過後才將 collection 標記為 complete。它不會啟動
Isaac Sim，也不會額外收集 episodes。若 episodes 缺漏或資料無效，仍會回報原始 validation
錯誤。

只有在 dataset collection 完成並通過 validator 後，才啟動訓練：

```bash
./uav bc-train --dataset bc_expert_cube --epochs 500
./uav bc-train --dataset bc_expert_cube --image-source fpv_rgb --epochs 100 \
  --encoder <fpv-ae-run>/best.pt
./uav bc-train --dataset bc_expert_cube --image-source fpv_depth --epochs 100 \
  --encoder <depth-ae-run>/best.pt
./uav bc-train --help
```

預設值為正式 dataset `artifacts/datasets/bc_expert_cylinder_v1`、既有 frozen encoder
`autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt`、learning rate `1e-3` 的 Adam、
batch size 64、最多 100 epochs、patience 12 和固定 seed。預設 image source 為 TOP。未指定
`--encoder` 時，BC 只讀取相同 dataset/source 已完成且相符的 AE provenance index，並驗證
其 summary、checkpoint metadata、hash、preprocessing、architecture 與 64D latent contract。
它不會 fallback 到其他 source 或 dataset。

FPV RGB 從 `samples.csv.image_path` 讀取。TOP RGB 與 FPV depth 則依據精確的
`episode_id`/`sample_id` identity 和 primary timestamp，從
`auxiliary.csv.observer_rgb_path` 與 `fpv_depth_path` 配對。程式會檢查 availability、
matched status、timestamp error 和檔案，不會 fallback 到其他 source。三種來源都使用完整的
`fixed_global_top` comparison cohort，因此相同 seed 會得到相同 deterministic split。稀疏的
legacy TOP episodes 會明確排除在此 comparison cohort 之外。

Split 依 episode 進行，比例約為 80/10/10，並具確定性。同一 episode 的 frames 不會分散到
train、validation 和 test。選定 validation 表現最佳的 checkpoint 後，才會載入 test split。
精確切分配置與排除紀錄會保存至 `split_manifest.json`。

每次 run 會寫入以下目錄：

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

`best.pt` 和 `last.pt` 包含 model 與 optimizer states、normalization、configuration、
random seed、dataset manifest reference/hash、完整 split、encoder path/hash，以及
observation/action contracts。Policy 訓練期間 encoder 會維持 eval mode 並停用 gradients。
成功的 run 也會更新 Git 忽略的 `artifacts/experiments/bc_baseline/latest.json`；evaluation
預設會使用此索引。

Offline metrics 包含各 action component 權重相同的 normalized action MSE，以及各 action
的 MSE、MAE 和 RMSE。它們回答的是：**BC 能否模仿保留測試集中的 expert actions？**這些
指標不能證明 UAV 能成功導航。

BC output 會自動寫入 `artifacts/experiments/bc/<dataset>/<source>/run_<timestamp>`。CLI
預設會啟動受管理的 TensorBoard server，並在成功訓練後持續執行直到按下 Ctrl+C。
`--no-tensorboard` 只停用 server，不會停用 event-file 記錄。TensorBoard 會記錄
`bc/train_action_loss` 和 `bc/validation_action_loss`（每個 epoch 各一筆）。重新載入
`best.pt` 後，另記錄一次 `bc/test_forward_rmse`、`bc/test_right_rmse` 和
`bc/test_yaw_rate_rmse`。啟動 TensorBoard：

之後仍可用以下命令重新開啟保留的 events：
`tensorboard --logdir <run-directory>/tensorboard`.

遠端 server 可使用 SSH tunnel，例如 `ssh -L 6006:localhost:6006 user@server`，再於本機
開啟 `http://localhost:6006`。正式 closed-loop runtime 接受 TOP RGB checkpoint contract。

## Closed-loop 評估

訓練產生 `best.pt` 後，執行：

```bash
./uav bc-eval --episodes 20
./uav bc-eval --help
```

可使用 `--checkpoint PATH` 指定 checkpoint。工具會載入 checkpoint 記錄的 encoder、驗證
其 SHA-256、沿用訓練時的 preprocessing 和 normalization，並啟動受管理的 Isaac Sim、
Pegasus、PX4 SITL 與 ROS 2 flight stack。

每次 rollout 中，`CONTROL SOURCE = BC_POLICY`。Evaluator 只會呼叫 BC policy 產生 actions；
不會呼叫環境中的 A* expert、不會混合 expert action，也不會把 privileged information 加入
policy observation。A* 僅在內部用來產生可到達的隨機場景，不是 control source。Collision
和越界終止都會回報為 collision failures。

Evaluation output 會寫入以下目錄：

```text
artifacts/evaluations/bc_flight/run_<timestamp>/
  metrics.json
  episode_<index>/result.json
  plots/
```

每個 episode 會記錄 seed、success、collision、timeout、終止原因、最小/最終 goal distance、
飛行時間、測量 path length、policy ownership、安全中止狀態與 blending 狀態。Aggregate
metrics 包含數量/比例，以及平均距離、時間和 path length。這些 closed-loop metrics 回答：
**BC 能否獨立飛到目標？**不可將較低的 offline MSE 解讀為導航成功。

## 範圍與目前風險

此基準流程不包含 encoder fine-tuning、RGB-D fusion、recurrent 或 attention modeling、
DAgger、GAIL、PPO 或 hyperparameter search。Closed-loop environment 使用確定性的固定高度
dynamics 和 Isaac-rendered camera；它可提供 policy-only 證據，但不能取代另行授權的
PX4/Pegasus learned-policy 飛行或硬體驗證。正式 Pegasus dataset 與此 evaluation renderer
之間的 domain shift 仍是明確風險。

產生的 datasets、checkpoints、metrics 和 plots 都由 Git 忽略。

## Closed-loop 時間紀錄與閱讀方式

`./uav bc-eval` 會在每回合資料夾自動保存以下診斷資料，沿用既有 node 和 evaluator，
不需要另外啟動 Python 診斷程式：

- `timing_summary.md`：中文報告。先看回合結果與首次故障，再看各階段的平均、P95、
  最大耗時與事件順序。原始錯誤文字保留英文以方便搜尋。
- `timing_summary.json`：相同觀測數據的結構化版本。
- `control_summary.md`：每回合優先閱讀的中文控制摘要，只列結果、新影像率、影像到 PX4、
  影像年齡、動作重送、最大 timer 間隔、ULog 是否可比較與下一個假說。
- `closed_loop_control_summary.md`：整次 closed-loop 的回合總覽。先讀這份，再依異常回合
  的連結開啟該回合的 `control_summary.md`。
- `timing/*.jsonl`：各 node 的原始時間事件，可依報告提供的檔名和行號回查。
- `px4_ulog/`：ULog、保存清單和 logger 原始輸出。Evaluator 會在 Pegasus 清除 temporary
  rootfs 前取得並保存檔案；保存失敗時會明確回報錯誤。

時間事件包含主機 monotonic、ROS 與 wall clock、來源訊息標記、發布與 callback 時間、
timer 間隔、判斷時的快取年齡、影像讀取與編碼、BC 推論耗時，以及 gate/streamer 狀態與
當時有效門檻。診斷不會修改控制、安全門檻或 PX4 參數；ULog 使用 logger 指令啟停。

**從發布到 callback 的耗時包含傳輸與排程，不能直接稱為 DDS 延遲。**快取年齡也不代表
傳輸耗時，而且 snapshot 可能包含未選用的控制來源。目前尚未量得影像實際擷取時間、DDS
內部排隊時間或 PX4 收到命令後的執行耗時，也尚未校準 PX4 boot clock 與主機時間。紀錄
本身會增加少量負載，但尚未透過對照實驗量化，因此單回合數值不足以作為固定門檻的依據。

只重新整理已保存的資料，不啟動模擬：

```bash
python scripts/diagnostics/summarize_bc_startup.py artifacts/evaluations/bc_flight/<run_directory>
```

### FPV 與 action 時間證據

FPV 的 `image_read` events 會分別記錄 pose update、annotator read 和 JPEG encoding。配對的
`publish` event 帶有 publication sequence 與 ROS message key。此 sequence **不是** renderer
frame ID。Capture time、render 完成時間，以及 pose update 是否套用於回傳影格，目前仍明確
標示為未知；不會增加 render 次數或改變 sensor rate。

BC 會記錄每次 inference 使用的 image、odometry 和 goal source keys，以及主機接收時間。每個
action 都有 ID、inference 完成時間與 repeated-action flag。Mux、gate 和 streamer 的
publications 只會在 diagnostic JSONL 保留所消耗的 input references。ROS messages 和 model
inputs 不變。報告只會為 BC 選取且 safe-to-forward 的 commands 連接這些 references，並回報
PX4 publication 時的 image receipt age（包括重複 actions）以及每個 action 的首次發布。這些
測量只到 host publisher API 為止，不代表 PX4 已收到。舊 recordings 缺少 references 時會維持
未連結，不會猜測配對結果。

報告使用選用的 `pyulog` 檢查已保存的 ULogs（在產生報告的 interpreter 執行
`python -m pip install pyulog`）。它會將可用的 setpoint、position/velocity、attitude 和
timesync topics 匯出成各 ULog 旁的 CSV，列出缺少的 topics，並在 JSON 保留 dropout 時間戳與
持續時間。Import 或 parse 失敗會顯示在 stderr 和報告中。ULog timestamps 不會自動等同主機
時間或實際執行時間；Logger 也可能降低 setpoints 的記錄頻率。未校準 clock 前，不會計算
host-to-PX4 delay。CSV 的 NaN 值保留 PX4 未設定欄位的表示方式。

啟用 timing 時，streamer 也會記錄 `/fmu/out/timesync_status` offsets、remote timestamps、
protocol、round-trip time 和 host clocks。若缺少 message support，報告會明確指出。沒有
publisher 就不會有證據；報告不會只因 timesync 可用，就宣稱 host/PX4 已完成校準或量得
control execution latency。

### 有界排程診斷

Control callbacks 會記錄進入、結束、wall duration、thread CPU duration 以及 PID/TID。若要在
已規劃的 targeted run 中額外擷取每個 node 前 30 秒的 Linux scheduler 和 I/O counters：

```bash
UAV_TIMING_SCHED_SECONDS=30 ./uav bc-eval --episodes 1 --checkpoint <same_checkpoint> --seed <same_seed>
```

此環境變數會由 runtime processes 繼承。額外 counter sampling 預設停用，並會自動停止。
Counters 在 timer entry 擷取，不會新增 worker 或 subprocess。`schedstat` 測量呼叫 thread
在 runqueue 的等待時間。Block I/O delay 以 kernel ticks 回報；process read/write bytes 是
輔助證據，不代表耗時。Counters 為零不能證明 kernel accounting 已啟用。缺少 `/proc` 證據時，
系統會明確記錄並停止後續 sampling。Callback wall time 減去 CPU time 仍包含 preemption 和
其他等待，因此不可標記為 I/O delay。診斷負載尚未透過 simulator 對照測試量化。

### 啟動前確認地面狀態

Supervisor 將缺少或過期的 land detection 視為未知，而不是 airborne。選擇 lifecycle control
前，啟動流程需要新鮮的 disarmed 與 landed 證據。地面狀態未知時會暫停啟動/復原，不發出
enable、mode、arming 或 reset requests；若已選取 source，則切換到 HOLD。既有 readiness
timeout、整體啟動期限、復原上限和 250 ms command timeout 都維持不變。明確 airborne 證據
會使流程中止；OFFBOARD request/history 仍會禁止 startup recovery。缺少地面證據或證據超過
1.5 秒時，status `landed` 為 `null`。

啟用 timing 時，control 和 status publishers 也會記錄 `publish_api_start` 與
`publish_api_end`，包含 topic、source key、wall time、thread CPU time 和成功狀態。既有
publication/lineage events 會保留。API duration 不包含 diagnostic writes，也不會測量 PX4
接收或執行時間。API duration 偏長代表 publish call 內有等待，但單靠此數值無法區分 DDS
locks/backpressure 和 CPU preemption。可使用前述有界 scheduler counters 調查；accounting
為零或不可用，不代表排程延遲為零。

Targeted regressions（請在專案測試環境手動執行）：

```bash
PYTHONPATH=ros2_ws/src/uav_px4_control:. python -m pytest ros2_ws/src/uav_px4_control/test/test_bc_startup_recovery.py ros2_ws/src/uav_px4_control/test/test_bc_flight.py ros2_ws/src/uav_px4_control/test/test_px4_output_gate.py
PYTHONPATH=ros2_ws/src/uav_px4_control:. python -m pytest tests/ml/test_bc_flight.py -k TimingReportTests
```
