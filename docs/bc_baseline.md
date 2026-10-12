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

### 可選影像來源組合

使用 `--image-sources` 指定一個以上的已登記來源；其清單必須與對應 AE checkpoint 完全
相同且順序一致：

```bash
./uav bc-train --dataset bc_expert_cube \
  --image-sources fpv_rgb fpv_depth --epochs 100

./uav bc-train --dataset bc_expert_cube \
  --image-sources fpv_rgb fpv_depth top --epochs 100
```

多影像 AE 將各來源 latent 融合為 64D 表示，所以 BC 仍使用既有 72D latent+state8 actor。
可以用 `--initialize-policy <single-source-best.pt>` 從相同 primary source 的單影像 BC
checkpoint 初始化 actor，再由所選影像組合續訓。BC checkpoint 固定記錄來源、順序和
preprocessing；飛行端依 checkpoint 訂閱所有來源，並在任一路影像缺失、過期或未同步時
停止送出新 policy action。多影像 TOP 要符合 1 ms 同步門檻；FPV RGB/depth 使用 100 ms。

舊有 `--image-source` 單來源指令仍可使用。

### 純 ResNet18 訓練與 64D encoder

`resnet-train` 直接使用指定資料集的影像、state8 和專家動作，從隨機權重訓練
ResNet18、64D 投影層及動作預測 head。訓練目標是三個 normalized actions 的 MSE；
此流程沒有 decoder，也不計算 reconstruction loss。沿用 episode split、最佳 validation
checkpoint、early stopping、TensorBoard 和 action-error plots。

```bash
./uav resnet-train --dataset cylinder_v3 --image-source top \
  --epochs 100 --no-tensorboard

./uav resnet-train --dataset cylinder_v4 --image-source top \
  --epochs 100 --no-tensorboard
```

`--dataset` 接受 datasets 下的名稱或完整路徑；每次 run 使用一個資料集。
預設 output 為 `experiments/resnet/<dataset>/<source>/run_<timestamp>`，也可指定
`--output`。影像沿用完整 128x72 preprocessing，encoder 內套用 ImageNet mean/std，
不下載 pretrained weights。state8 的 normalization 只由 train split 計算；訓練時的
64D latent 不使用會隨 encoder 權重變動的資料統計。

最佳 encoder 寫入 `resnet18_encoder.pt`；`resnet18_encoder_last.pt` 保存最後一個 epoch。
兩者均包含 ResNet18 和 64D 投影層，不包含 AE decoder。`best.pt`、`last.pt` 保存相符的
動作 head、normalization 和 encoder 路徑/hash，供離線或既有 evaluator 使用。
純 ResNet run 不會更新 BC 的全域 latest pointer。

若要維持先訓練 ResNet，再固定 encoder 訓練 BC 的兩階段流程：

```bash
./uav bc-train --dataset cylinder_v3 --image-source top \
  --encoder experiments/resnet/cylinder_v3/top/run_<timestamp>/resnet18_encoder.pt \
  --epochs 100 --no-tensorboard
```

BC 會使用固定的 `latent64 + state8`，輸入維度仍為 72。
請讓兩階段使用相同 dataset、image source、seed 和 episode subset。
BC 的 `--encoder` 同時接受訓練完成的 64D AE 或純 ResNet checkpoint。

### 總訓練時長

AE、BC 和純 ResNet 的 completed runs 都在 `summary.json` 和 `training_config.json`
記錄 `training_duration_s`（秒）；BC/ResNet 也寫入 `metrics.json`。此時長以 monotonic
clock 計算，包含資料準備、訓練、評估和圖表產生，不包含訓練結束後等待 TensorBoard
或 Ctrl+C 的時間。完成訊息會直接顯示總秒數。
舊版 `python -m uav_ml.train_bc` 也會在 checkpoint 的
`training_metadata.training_duration_s` 和 terminal 結果記錄時長。

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
  --output experiments/bc/bc_expert_cube/top/resnet18/run_01

./uav bc-eval --image-source top_rgb --episodes 1 --visible \
  --checkpoint experiments/bc/bc_expert_cube/top/resnet18/run_01/best.pt
```

基本 BC 參數為 `--epochs`、`--batch-size` 和 `--learning-rate`。使用
`--no-tensorboard` 會在訓練後結束，但仍會保存 metrics 和 plots；省略此參數則會讓受管理的
TensorBoard server 持續執行，直到按下 Ctrl+C。`--encode-batch-size` 可獨立控制 encoder
記憶體用量（預設 128）。每次 run 請使用新的 output directory；現有 experiments 不會被
覆寫。未指定 `--output` 時，ResNet runs 會自動寫入
`experiments/bc/<dataset>/<image_source>/resnet18/run_<timestamp>`；預設 AE
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
./uav expert-validate --dataset datasets/bc_expert_cylinder_v2 --finalize
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

預設值為正式 dataset `datasets/bc_expert_cylinder_v1`、既有 frozen encoder
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
experiments/bc/<dataset>/<source>/run_<timestamp>/
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
成功的 run 也會更新 Git 忽略的 `experiments/bc/latest.json`；evaluation
預設會比較新目錄內各個已完成 run 的索引。舊的 `artifacts/experiments/bc_baseline/`
保留原位，不參與這次搬移。

Offline metrics 包含各 action component 權重相同的 normalized action MSE，以及各 action
的 MSE、MAE 和 RMSE。它們回答的是：**BC 能否模仿保留測試集中的 expert actions？**這些
指標不能證明 UAV 能成功導航。

BC output 會自動寫入 `experiments/bc/<dataset>/<source>/run_<timestamp>`。CLI
預設會啟動受管理的 TensorBoard server，並在成功訓練後持續執行直到按下 Ctrl+C。
`--no-tensorboard` 只停用 server，不會停用 event-file 記錄。TensorBoard 會記錄
`bc/train_action_loss` 和 `bc/validation_action_loss`（每個 epoch 各一筆）。重新載入
`best.pt` 後，另記錄一次 `bc/test_forward_rmse`、`bc/test_right_rmse` 和
`bc/test_yaw_rate_rmse`。啟動 TensorBoard：

之後仍可用以下命令重新開啟保留的 events：
`tensorboard --logdir <run-directory>/tensorboard`.

遠端 server 可使用 SSH tunnel，例如 `ssh -L 6006:localhost:6006 user@server`，再於本機
開啟 `http://localhost:6006`。Closed-loop runtime 依 checkpoint 記錄的來源組合訂閱影像；
啟動時 `--image-source` 指定 checkpoint 的 primary source，其餘必需來源會由 checkpoint 載入。

## Closed-loop 評估

訓練產生 `best.pt` 後，執行：

```bash
./uav bc-eval --episodes 20
./uav bc-eval --help
```

可使用 `--checkpoint PATH` 指定 checkpoint。工具會載入 checkpoint 記錄的 encoder、驗證
其 SHA-256、沿用訓練時的 preprocessing 和 normalization，並啟動受管理的 Isaac Sim、
Pegasus、PX4 SITL 與 ROS 2 flight stack。

未指定 `--checkpoint` 時，會比較 `experiments/bc/` 中各 `latest.json` 的
`updated_utc`，選取最近完成且 checkpoint 仍存在的 BC 訓練，涵蓋各 dataset、AE、ResNet18
與影像來源。`./uav bc-train` 成功完成後會原子更新全域的
`experiments/bc/latest.json`，即使訓練使用自訂 `--output` 也能供下次評估選取。
未完成的訓練不會取代預設模型；不可用的 pointer 會在 terminal 顯示原因。
未指定 `--image-source` 時，會沿用所選 checkpoint 的主要影像來源與其餘必要來源；
明確指定來源時仍會驗證是否符合 checkpoint。啟動時 terminal 會顯示 checkpoint 路徑與
影像來源，方便確認當次使用的模型。

每次 rollout 中，`CONTROL SOURCE = BC_POLICY`。Evaluator 只會呼叫 BC policy 產生 actions；
不會呼叫環境中的 A* expert、不會混合 expert action，也不會把 privileged information 加入
policy observation。A* 僅在內部用來產生可到達的隨機場景，不是 control source。Collision
和越界終止都會回報為 collision failures。

正式 BC flight evaluation 依 checkpoint 實際載入的特徵擷取模型決定預設目錄，
使用台北時區的 `月份日期T小時分鐘Y西元年`：

```text
evaluations/bc/AE_1010T1530Y2026/
  summary.json
  episode_<index>/result.json
  trajectory_plots/
  summary_plots/
  loss/bc.png
  loss/ae.png
  loss/loss_summary.json
```

ResNet18 則使用 `evaluations/bc/ResNet18_1010T1530Y2026/`，包含
`RgbAutoencoderV0`、`MultiImageAutoencoderV0` 的 AE 都使用 `AE_` 前綴。
`--output PATH` 仍可指定完整輸出目錄；`--nas-root` 發布到 NAS 的 `evaluations/bc/`。
既有目錄不會自動搬移，同一分鐘同類模型的同名目錄也不會被覆寫。
專案根目錄的 `evaluations/` 已由 `.gitignore` 忽略，所有評估產物都不會加入 Git。

`loss/loss_summary.json` 記錄所選 BC 與 encoder checkpoint 的絕對路徑、checkpoint epoch、
各自的最低 training loss／validation loss epoch 與 loss，以及最大 epochs、learning rate、
batch size、early stopping patience、minimum epochs、seed、實際 epochs 和訓練總時長。
Train 與 validation 的最佳 epoch 分開從該模型的歷史紀錄計算；驗證最佳 epoch 不代表
training loss 的最低點。曲線和摘要來源都是所選 checkpoint 同目錄的訓練產物，不使用 latest。

BC／AE 後續訓練的 `summary.json` 和 `training_config.json` 會保存 `training_duration_s`，
單位為秒，涵蓋資料準備、訓練與訓練產物生成。舊訓練沒有紀錄時，loss 摘要會使用 `null`
並列入 `missing_fields`；缺少曲線檔時可由 CSV 重新繪圖。
ResNet18 使用凍結的 `IMAGENET1K_V1` 權重，沒有本專案的 encoder loss 或訓練時長，
摘要標示 `pretrained_frozen`，不產生 encoder loss 圖；BC 的 `loss/bc.png` 照常輸出。

每個 episode 會記錄 seed、success、collision、timeout、終止原因、最小/最終 goal distance、
飛行時間、測量 path length、policy ownership、安全中止狀態與 blending 狀態。Aggregate
metrics 包含數量/比例，以及平均距離、時間和 path length。這些 closed-loop metrics 回答：
**BC 能否獨立飛到目標？**不可將較低的 offline MSE 解讀為導航成功。

正式 `./uav bc-eval` 的 `summary.json` 會保存 `total_test_duration_s`，單位為秒。
此欄位使用 monotonic clock，從 checkpoint 驗證前計時，直到最後一回合的 runtime 清理結束；
包含模擬啟動、飛行、診斷影片產生與清理，不包含最後的統計繪圖和 NAS 傳輸。
中斷或啟動失敗時，只要已建立本次輸出目錄，仍會保存截至清理結束的實際耗時。

`videos/episode_XXXXXX.mp4` 使用四項資訊同框的 2560×1440 版面：左側為 BC 成功推論
實際使用的主要來源影像，右上為 Isaac TOP 相機，右下為機體座標方向指針與
最近十步動作，格式為 `step7: [0.50, 0.20, -0.10]`，依舊到新由上而下排列，最新一步
高亮；不足十步時顯示已有紀錄，第十一筆起移除最舊一筆。step 保留原始成功推論編號。
三個數值依序為 forward/right/yaw rate（m/s、m/s、rad/s），沿用 runtime 的動作
裁切與縮放；指針表示平移命令方向，yaw rate 另外以數值顯示，並非實測飛行速度。
指針預設為實心三角形；修改 `uav_ml/tools/bc_flight_video.py` 的 `POINTER_STYLE` 可切換
`arrow`（細線箭頭）、`triangle`（實心三角形）、`chevron`（V 形導航尖頭）或
`needle`（雙色羅盤指針），不改變動作或模型輸入。
TOP 依主要輸入影像的時間戳在 1 ms 內配對；缺少同期 TOP 時明確顯示缺圖，不等待或
影響 policy readiness。錄影 TOP 使用同一 ObserverCamera 的獨立 1024×1024 輸出，JPEG
quality 95、4:4:4，僅在 BC 錄影時啟用。模型仍使用既有 FPV 320×180／TOP 640×640、
JPEG quality 85 與既有前處理。指針與文字以 2 倍 supersampling 抗鋸齒，TOP 顯示使用
Lanczos，H.264 編碼使用 CRF 16。舊 runtime 沒有高解析度 TOP 時，可沿用同期的模型
TOP；舊影像不會因提高影片解析度而增加原始細節。多影像模型左側顯示
主要來源；FPV depth 僅在影片中轉為 0–30 m 灰階。影片標示實際來源，不將 TOP 標成 FPV。
旁邊的 JSON 保存版面、指針樣式、動作歷史長度、動作單位與具備 TOP 畫面的 frame 數；舊錄影來源缺少 TOP 時
仍可合成，舊 MP4 不會自動重製。

`trajectory_plots/` 保留每航次的 `trajectory_1.png` 至 `trajectory_20.png`，另外依固定
航次範圍新增四條軌跡的合併圖：`trajectory_001-004.png`、`trajectory_005-008.png`，
依此類推。缺少有效 trace 的航次會略過，不會移動其他航次的分組。合併圖中，timeout 為
藍色實線、success 為綠色實線、collision 為紅色實線、control issue 為紫色虛線；
同組同結果使用不同深淺，圖例標示航次與結果。越界沿用灰色實線。

`summary_plots/summary.png` 只顯示 Success、Timeout、Collision，Y 軸次數使用整數。
百分比仍以全部航次為分母；其他終止原因保留在 JSON 與 `flight_status_by_episode.png`。
自行調整字體時，修改 `uav_ml/tools/bc_flight_plotting.py` 頂端的下列常數，數值單位為 pt：

| Constant | Text | Default |
| --- | --- | --- |
| `SUMMARY_TITLE_FONT_SIZE` | 圖表標題 | 14 |
| `SUMMARY_LABEL_FONT_SIZE` | Y 軸標籤 Episodes | 12 |
| `SUMMARY_TICK_FONT_SIZE` | X/Y 軸刻度與結果名稱 | 11 |
| `SUMMARY_COUNT_FONT_SIZE` | 柱上次數與百分比 | 11 |

例如將 `SUMMARY_TITLE_FONT_SIZE = 14` 改成 `18`，再執行以下指令重新產圖：

```bash
./uav bc-eval-plot evaluations/bc/AE_1010T1530Y2026
```

重新產圖會更新既有 `summary.json` 的圖表路徑，保留原始測試時長；舊測試沒有實際耗時
紀錄時，不會由飛行時長推算或補造此欄位。

## 範圍與目前風險

此基準流程不包含 encoder fine-tuning、RGB-D fusion、recurrent 或 attention modeling、
DAgger、GAIL、PPO 或 hyperparameter search。Closed-loop environment 使用確定性的固定高度
dynamics 和 Isaac-rendered camera；它可提供 policy-only 證據，但不能取代另行授權的
PX4/Pegasus learned-policy 飛行或硬體驗證。正式 Pegasus dataset 與此 evaluation renderer
之間的 domain shift 仍是明確風險。

產生的 datasets、checkpoints、metrics 和 plots 都由 Git 忽略。

## Closed-loop 時間紀錄與閱讀方式

`./uav bc-eval` 預設不記錄 timing 或 PX4 ULog，也不產生下列診斷報告。
結果、軌跡、影片、圖表及原始 terminal logs 仍照常保存；既有 artifacts 不會被刪除。
需要排錯時加入 `--diagnosis`，沿用既有 node 和 evaluator，不需要另外啟動診斷程式：

```bash
./uav bc-eval --diagnosis --episodes 1 --checkpoint <same_checkpoint> --seed <same_seed>
```

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
當時有效門檻。診斷不會修改控制或安全門檻。評估時會以 `SDLOG_MODE=-1` 關閉 PX4
自動 logger；只有 `--diagnosis` 會使用 logger 指令啟停並保存 ULog。

**從發布到 callback 的耗時包含傳輸與排程，不能直接稱為 DDS 延遲。**快取年齡也不代表
傳輸耗時，而且 snapshot 可能包含未選用的控制來源。目前尚未量得影像實際擷取時間、DDS
內部排隊時間或 PX4 收到命令後的執行耗時，也尚未校準 PX4 boot clock 與主機時間。紀錄
本身會增加少量負載，但尚未透過對照實驗量化，因此單回合數值不足以作為固定門檻的依據。

只重新整理已保存的資料，不啟動模擬：

```bash
python scripts/diagnostics/summarize_bc_startup.py evaluations/bc/<run_directory>
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
UAV_TIMING_SCHED_SECONDS=30 ./uav bc-eval --diagnosis --episodes 1 --checkpoint <same_checkpoint> --seed <same_seed>
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
