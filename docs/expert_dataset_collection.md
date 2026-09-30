# Expert Dataset Collection 工具

正式 collector 將已驗證的 Isaac → ROS 2 → A* → PX4 飛行流程包裝成可續跑、單一命令
啟動的資料集工作流程。資料集位置為：

```text
artifacts/datasets/bc_expert_highrise_v1/
```

整個 `artifacts/` 產生資料目錄都由 Git 忽略。影像、depth PNG、飛行證據、manifests、
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
終點 `(3, 5)`、恰好八棟高樓、固定的 width/depth/height/yaw 範圍、0.50 m 最小間距、
兩個保證阻擋直線路徑的障礙物，以及 canonical lighting。Collector 會在飛行開始前驗證
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

Per-episode 與 aggregate validation 會檢查 seed 唯一、canonical 八棟高樓/兩個阻擋物場景、
建築邊界與間距、lighting、JPEG 可讀性與尺寸、uint16 PNG depth、timestamp 單調性、同步
容許值、rejection accounting、stream 數量、A* path metadata、終止結果與安全失敗證據、
sampling rate、磁碟統計及 Visual QA 頻率。每個 accepted sample 都會解碼，並通過目前的
preprocessing/encoder，以重建有限的 64D latent、72D observation 與 3D expert target。

此工具不會啟動 BC 或 PPO 訓練。
