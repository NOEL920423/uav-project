# 架構導覽

日常研究只沿以下兩條流程進行。訓練在呼叫端的 Python 環境執行；ROS 2 使用 Ubuntu 上的 Jazzy 環境。

```text
expert-collect -> ae-train 或 resnet-train -> bc-train -> bc-eval
```

## 收集與訓練

```text
./uav expert-collect
  -> uav_ml/tools/expert_collect.py
  -> uav_ml/tools/persistent_runtime.py
  -> isaac/runtime/bootstrap.py + environment.py
  -> ./uav expert-run-episode
  -> px4_sitl_flight.launch.py
  -> expert_dataset_recorder -> datasets/<dataset>

./uav ae-train     -> uav_ml/train_autoencoder.py
./uav resnet-train -> uav_ml/tools/bc_baseline.py 的 ResNet encoder 訓練
./uav bc-train     -> uav_ml/tools/bc_baseline.py -> BC checkpoint
```

AE、ResNet、BC 的 split、normalization、影像來源與 checkpoint provenance 沿用各次實驗設定。
共用 random seed 與裝置選擇函式位於 `uav_ml/training/__init__.py`。

## 閉環測試

```text
./uav bc-eval
  -> uav_ml/tools/bc_flight_evaluation.py
  -> Isaac Sim + Pegasus + PX4 SITL + XRCE
  -> ./uav bc-flight-run -> bc_flight.launch.py

影像 + PX4 odometry + goal
  -> bc_policy -> control_mux -> px4_mapping_gate
  -> px4_setpoint_streamer -> PX4 -> Pegasus vehicle

bc_flight_supervisor -> 起飛、來源切換、HOLD、降落
bc_episode_monitor   -> 終止條件、result.json、軌跡
```

`bc-eval --diagnosis` 額外保存 timing、ULog 與控制摘要。`Control issue` 是
`runtime_failure` 的圖表標籤，並非根因；已知證據與限制見 [control_issue.md](control_issue.md)。

## 五個 ROS 2 套件

| 套件 | 職責 |
|---|---|
| `uav_interfaces` | 正式流程共用的 messages 與 services |
| `uav_scene_bridge` | Isaac runtime 與 coherent scene snapshots |
| `uav_navigation` | Expert 的 A*、B-spline、軌跡參數化、follower |
| `uav_px4_control` | Source mux、PX4 邊界、expert／BC 生命週期與診斷 |
| `uav_data_recorder` | 場景準備與同步 expert dataset 記錄 |

## 正式 launch 與 nodes

僅保留 `ros2_ws/src/uav_px4_control/launch/` 下兩個 launch。

| 使用範圍 | Nodes |
|---|---|
| 兩者共用 | `scene_bridge_node`, `px4_odometry_bridge_node`, `control_mux_node`, `px4_live_telemetry_adapter`, `px4_mapping_gate_node`, `px4_setpoint_streamer_node`, `px4_vehicle_command_owner_node` |
| `px4_sitl_flight.launch.py`：收集 | `astar_planner_node`, `trajectory_parameterizer_node`, `trajectory_follower_node`, `expert_dataset_recorder`, `px4_sitl_flight_supervisor_node`, `px4_sitl_flight_monitor` |
| `bc_flight.launch.py`：測試 | `bc_policy_node`, `bc_flight_supervisor_node`, `bc_episode_monitor_node` |

`episode_scene_client`、`px4_sitl_doctor`、`px4_generation_probe` 與
`runtime_smoke_lifecycle_client` 由 orchestrator 按需啟動，負責 scene、readiness、跨回合 reset。
它們仍是正式流程的相依項。

## 保留與移除的範圍

保留正式流程使用的純演算法、契約／安全測試與其 fixtures、故障診斷、資料驗證、影片／繪圖及 NAS 工具。
`diagnostics/` 中部分純 Python fixtures 僅供測試使用，不會啟動額外 ROS graph。
控制來源的既有識別字與 message 格式保留相容性；日常入口只提供自動 expert 與 BC 流程。

已移除 Phase 1 scaffold 套件／nodes、獨立離線 launch／harness、實體搖桿入口、
舊版 NPZ 四軸 BC trainer／model／inference、固定高度 city／PPO 支線及單回合 overfit 指令。
相關安裝入口、失效文件和舊架構圖同步移除。

Dataset、所有 checkpoint、歷史 logs、失敗結果與其他研究 artifacts 均保留。
精簡只降低維護範圍，不代表既有閉環 control issue 已修復。

Isaac Sim、PX4 與 Pegasus 維持外部安裝。專案內 `pegasus_test/PegasusSimulator`
的 144 個 source／asset files 已逐檔驗證與 `/home/noel_614420090/PegasusSimulator`
相同，重複副本改為連到該 checkout 的 symlink；不另外維護第三方程式副本。

## 本次精簡驗證

- 五個 ROS 套件重新建置成功，保留的 20 個 console entry points 均可 import。
- ML／runtime：163 tests 通過；ROS 功能測試：375 通過、22 跳過；CLI 檢查通過。
- 現有 BC checkpoint 與其 encoder 可載入；未啟動 training、collection 或 Isaac 飛行。
- 當時完整 ROS workspace 測試仍有 recorder、navigation、PX4 control 三個既有 lint suites 失敗。
  功能測試結果與 lint 分開回報；未放寬 lint 設定或移除其測試。
