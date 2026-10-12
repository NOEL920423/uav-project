# Expert Dataset Collection 工具

正式 collector 將已驗證的 Isaac → ROS 2 → A* → PX4 飛行流程包裝成可續跑、單一命令
啟動的資料集工作流程。資料集位置為：

```text
datasets/bc_expert_highrise_v1/
```

整個 `datasets/` 產生資料目錄都由 Git 忽略。影像、depth PNG、飛行證據、manifests、
runtime logs 和 contact sheets 都不得提交至版本控制。

## 指令

將 100 筆通過驗證的成功 episodes 收集到新的 dataset：

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube
```

若收集過程中斷，可接續執行直到同一資料集達到 accepted 數量目標：

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube --resume
```

`--episodes` 指整個 dataset 中 accepted successful episodes 的目標數量，不是 seed
嘗試次數。既有 accepted episodes 也會計入目標。例如資料集已有 8 筆 accepted episodes，
以下指令會再收集 92 筆：

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube --resume
```

每次被拒絕的嘗試都會改用下一個 canonical seed，並保留在只能附加的 attempt history。
當 accepted 目標達成或總嘗試次數到達上限時，collection 會停止。預設上限為
`ceil(1.5 * --episodes)`，也可以明確指定：

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube \
  --max-attempts 150 --resume
```

已完成的 episode 目錄不會被覆寫。接續中斷的嘗試前，未完成的目錄會移至 Git 忽略的
recovery log。Episode IDs 至少保留六位數，超過 `episode_999999` 後會自然增加位數。

查看指令選項，或驗證既有的已完成 collection：

```bash
./uav expert-collect --help
./uav expert-validate --episodes 100
```

`--dry-run` 是離線 developer fixture。它會在 `run_logs/` 下測試 lifecycle、progress、
manifest 和 seed 邏輯，不會啟動 Isaac、ROS、PX4，也不會建立正式 dataset。

## 固定研究規約

每個正式 episode 都使用 `isaac/runtime/environment.py` 中的 normal mode：起點 `(0, 0)`、
終點 `(3, 5)`、兩個圓柱障礙物、0.50 m 最小間距，以及 canonical lighting。
使用 `--scene v3` 依 seed 生成障礙物，或 `--scene v4` 使用固定配置；未指定時維持 v4。
Collector 會在飛行開始前驗證
純確定性場景；aggregate validator 也會再次檢查記錄下來的場景。

Collector 不負責設定 camera geometry，而是啟動既有的 canonical sensor runtime，使用：

- FPV RGB：320×180 JPEG、quality 85，主要資料流約 5 Hz。
- FPV Depth：原始 uint16-millimetre PNG，auxiliary stream 約 5 Hz。
- Observer RGB：正式 TOP 解析度與實際涵蓋範圍一併在
  `isaac/runtime/formal_expert_sensor_contract.py` 設定，auxiliary stream 約 5 Hz。

BC V1 sample 格式維持不變：目前的 FPV preprocessing 和 frozen encoder 產生 64 個 latent
values；body velocity (2)、body-frame goal direction (2)、normalized distance (1) 與
previous normalized action (3) 組成 72D observation。Target 維持為正規化的
`[v_forward, v_right, yaw_rate]`，並沿用現有 action limits。

## 自動化生命週期與失敗處理

每個預定 seed 都會依序驗證場景、啟動隔離的 Isaac/PX4/XRCE runtime、確認已落地且解除
armed 狀態、套用場景、執行有限且受保護的 ASTAR_EXPERT 飛行、記錄同步資料流、降落、
附加安全終止證據、驗證 episode、清理所有自有 process groups，然後自動進入下一筆。

一般任務失敗（collision/tracking、場景受阻、安全的 A*/goal failure、episode 結構驗證失敗，
或排程的 visual QA artifact 失敗）會完成收尾並記為 rejected attempts，不會阻止後續 seeds。
Episode 目錄會保留，`rejected_attempts/attempt_XXXXXX.json` 會索引 seed、類別、原因、
episode/flight/validation 證據與 runtime log。Recorder/evidence 遺失、終止狀態不安全、檔案
系統輸出損毀、runtime readiness 失敗、內部 exception 或 process ownership 遺失都屬於
infrastructure failure；batch 會中止並保留可續跑的 manifest。

影像亮度、影像 dynamic range 與觀察到的 sample/sensor rates 目前會記錄為品質警告，不會
作為拒收 collection 的門檻。Episode 仍需包含成功且安全降落的飛行，以及結構正確、已同步、
影像可解碼且 observations/actions 有限的 samples。每個 episode 的警告儲存在
`validation.json`；aggregate warning 數量會記錄於 `collection_validation.json`，供日後檢討。

Planner readiness 同時接受已獨立驗證的 B-spline，以及 planner 完成碰撞檢查的
`ASTAR_FALLBACK` final path。因此 B-spline 被拒絕時，不會丟棄有效的 A* route。正式
takeoff supervisor 的 0.25 m 高度容許值也與 trajectory follower 的 0.25 m 終止位置
容許值一致，避免已穩定的 takeoff trajectory 仍略低於任務轉換邊界。

`collection_manifest.json` 是 resume 的唯一資料來源，並完整稽核 accepted 與 rejected
attempts。`dataset_manifest.json` 是提供給 BC 的 accepted manifest，只包含 accepted
episode IDs。`collection_summary.json` 記錄 requested、attempted、accepted、rejected、
rejection 類別、infrastructure failures 與完成狀態。

每個 episode 的 `episode.json` 會在 `scene_configuration.environment_parameters` 保存
環境參數快照，包含隨機/固定障礙物數量、場地邊界、地板與牆壁尺寸和顏色、起點與終點標記
尺寸、光源強度與曝光、Pegasus 環境光與 RTX 陰影/AO 開關，以及材質參數。每個障礙物的
實際形狀、位置、半徑、寬度/深度基準、高度、朝向和放置方式則保存在同一份
`scene_configuration.obstacles` 清單。`validation.json` 另記錄收集影像的亮度與動態範圍，
用來區分場景光照設定和實際影像亮度。

## 進度與 Visual QA

進度輸出由 recorder 每秒只寫入一次的 `progress.json` snapshot 驅動，再由 orchestration
process 讀取。它不會訂閱或延遲 sensor callbacks。Terminal 會顯示 episode/total、百分比、
seed、飛行狀態、成功/失敗數、目前與累計 accepted samples、拒絕數、資料集大小、經過時間、
ETA 和簡短狀態轉換。

每完成 20 個 episodes，工具會從最近一次成功 episode 製作 3×3 contact sheet：

```text
FPV 起始 / 飛行中 / 接近終點
Observer 起始 / 飛行中 / 接近終點
Depth 起始 / 飛行中 / 接近終點
```

Contact sheets 和其 source-path JSON 存放在 dataset 內的 `visual_qa/`。QA 產生失敗會
留下記錄，不會中斷安全的飛行收集；但缺少排定的 contact sheets 會使最終 aggregate
validation 拒絕該 collection。

## 驗證範圍

Per-episode 與 aggregate validation 會檢查 seed 唯一、所選版本的兩個阻擋物場景、
建築邊界與間距、lighting、JPEG 可讀性與尺寸、uint16 PNG depth、timestamp 單調性、同步
容許值、rejection accounting、stream 數量、A* path metadata、終止結果與安全失敗證據、
sampling rate、磁碟統計及 Visual QA 頻率。每個 accepted sample 都會解碼，並通過目前的
preprocessing/encoder，以重建有限的 64D latent、72D observation 與 3D expert target。

此工具不會啟動 BC 或 PPO 訓練。

## v3 / v4 場景切換與交叉測試

`config/scenes/v3.json` 與 `config/scenes/v4.json` 是獨立保存的場景設定。
v3 的生成器還原自 Git revision `d53c991`，並比對過去 v3 資料的完整障礙物記錄。
兩個版本都使用 X `[-5, 5]`、Y `[-2, 7]`、起點 `(0, 0, 0)`、目標 `(3, 5, 1.5)`、
兩個直線 blocker、0.50 m 最小間距及 intensity 800 的白色 DomeLight。

| 參數 | v3 | v4 |
|---|---|---|
| 障礙物生成 | episode seed 決定位置、尺寸、yaw 與裝飾 | 固定位置、尺寸、yaw 與裝飾 seed 0 |
| 額外隨機障礙物 | 0 | 0 |
| 第一個 blocker | width/depth basis 0.72 m，高 5.2 m；位置沿直線比例 0.34–0.38，加橫向 jitter ±0.08 m | `(1.08, 1.80)`，width/depth basis 0.72 m，高 5.2 m，yaw 0 |
| 第二個 blocker | width/depth basis 0.56–0.72 m，高 3.2–5.2 m；位置沿直線比例 0.62–0.66，加橫向 jitter ±0.08 m | `(1.92, 3.20)`，width/depth basis 0.64 m，高 4.0 m，yaw 0 |
| yaw 範圍 | −35° 至 35° | 0° |

歷史 v3 的每個 `episode.json` 已保存實際障礙物、牆壁、placement contract 與光源，
但沒有完整的 `environment_parameters` 或資料集層級設定 JSON。外觀設定由當時的 Git
程式確認；新紀錄會保存完整設定，不會覆寫或補造歷史 episode 資料。

### 蒐集指定場景

場景版本由 `--scene` 決定，與 `--dataset` 名稱獨立。使用新名稱避免覆寫既有資料集：

```bash
./uav expert-collect --scene v3 --episodes 100 --dataset cylinder_v3_new
./uav expert-collect --scene v4 --episodes 100 --dataset cylinder_v4_new
./uav expert-collect --scene v3 --episodes 100 --dataset cylinder_v3_new --resume
```

`--episodes` 的 accepted 目標語意不變。`expert-collect-publish` 與
`expert-scene-preview` 也接受 `--scene v3|v4`。
`--resume` 會在修改 attempt history 前核對設定 JSON；不同版本或不同參數會直接拒絕。
對於沒有設定快照的舊資料集，會先核對所有已有 episode 的場景，才建立資料集層級快照。

新輸出包含：

```text
datasets/<dataset>/scene_parameters.json
datasets/<dataset>/collection_manifest.json
datasets/<dataset>/episode_XXXXXX/episode.json
run_logs/expert-collection_*/episode_XXXXXX/expected_scene.json
run_logs/expert-collection_*/episode_XXXXXX/scene.json
```

`scene_parameters.json` 保存版本、完整生成與外觀參數，以及正規化 JSON 內容的 SHA-256。
`episode.json` 的 `scene_configuration` 保存相同設定、實際幾何、episode seed、版本與摘要。
`expected_scene.json` 是飛行前預期場景；`scene.json` 是 Isaac 實際回報的場景。
離線 `--dry-run` 只生成設定與預期場景，不會產生 Isaac 回報。

### 分別訓練與四組閉循環測試

已有 `datasets/cylinder_v3` 與 `datasets/cylinder_v4` 時可以直接訓練，不必重新蒐集。
下列命令示範原有的 TOP Autoencoder + BC 流程；epochs 是示範值，依實驗需求設定。
訓練時使用 `--dataset` 選資料來源；測試時用 `--checkpoint` 選模型、`--scene` 選場景。
BC 會選擇同一 dataset/image source 的 Autoencoder checkpoint。

```bash
./uav ae-train --dataset cylinder_v3 --image-source top --epochs 20
./uav bc-train --dataset cylinder_v3 --image-source top --epochs 100 \
  --output experiments/bc/cylinder_v3/top_rgb/scene_transfer
./uav ae-train --dataset cylinder_v4 --image-source top --epochs 20
./uav bc-train --dataset cylinder_v4 --image-source top --epochs 100 \
  --output experiments/bc/cylinder_v4/top_rgb/scene_transfer

./uav eval --scene v3 --checkpoint experiments/bc/cylinder_v3/top_rgb/scene_transfer/best.pt \
  --episodes 1 --seed 900000 --headless --output evaluations/bc/v3_train_v3_test
./uav eval --scene v4 --checkpoint experiments/bc/cylinder_v3/top_rgb/scene_transfer/best.pt \
  --episodes 1 --seed 900000 --headless --output evaluations/bc/v3_train_v4_test
./uav eval --scene v3 --checkpoint experiments/bc/cylinder_v4/top_rgb/scene_transfer/best.pt \
  --episodes 1 --seed 900000 --headless --output evaluations/bc/v4_train_v3_test
./uav eval --scene v4 --checkpoint experiments/bc/cylinder_v4/top_rgb/scene_transfer/best.pt \
  --episodes 1 --seed 900000 --headless --output evaluations/bc/v4_train_v4_test
```

`eval` 是 `bc-eval` 的別名，兩者接受相同選項。`--episodes 1` 是一次飛行；
一組評估要飛行多次時，請自行設定 episode 數量。四組使用相同的測試 seed 序列可比較場景
差異；v4 幾何不因 seed 改變。若使用新蒐集資料，將訓練與 checkpoint 路徑中的 dataset
名稱換成 `cylinder_v3_new`、`cylinder_v4_new`。重做實驗時請換新的 `--output` 目錄。
可以將 `--headless` 改為 `--visible` 開啟 WebRTC 預覽。

每組 evaluation 在根目錄保存 `scene_parameters.json` 與 `summary.json`，每個 episode
保存 `scene_status.json`、`scene.json`、`result.json`。Isaac 必須回覆相同場景版本與設定
SHA-256 才會啟動 BC 飛行；結果同時記錄模型 checkpoint SHA-256 與場景設定 SHA-256。
