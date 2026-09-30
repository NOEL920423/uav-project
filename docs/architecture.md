# 架構導覽

本文件只介紹正式工作流程：

```text
expert-collect -> ae-train -> bc-train -> bc-eval
```

## 正式訓練流程

```text
./uav expert-collect
  -> uav_ml.tools.expert_collect
  -> uav_ml.tools.persistent_runtime
  -> isaac/runtime/bootstrap.py
     (Isaac Sim + Pegasus + PX4 SITL + runtime_bridge)
  -> ./uav expert-run-episode
  -> px4_sitl_flight.launch.py
  -> expert_dataset_recorder
  -> artifacts/datasets/<dataset>

./uav ae-train
  -> uav_ml.train_autoencoder
  -> RGB episode dataset -> RgbAutoencoderV0 checkpoint

./uav bc-train
  -> uav_ml.tools.bc_baseline
  -> frozen AE + expert dataset -> LatentBcPolicy checkpoint
```

`ae-train` 和 `bc-train` 不依賴 ROS、Isaac 或 PX4。

## 正式 BC closed-loop 流程

```text
./uav bc-eval
  -> uav_ml.tools.bc_flight_evaluation
  -> isaac/runtime/bootstrap.py
     (Isaac Sim + Pegasus + PX4 SITL + runtime_bridge)
  -> ./uav bc-flight-run
  -> bc_flight.launch.py

TOP RGB + PX4 odometry + scene goal
  -> bc_policy
  -> control_mux
  -> px4_mapping_gate
  -> px4_setpoint_streamer
  -> PX4 /fmu/in trajectory setpoint + offboard mode
  -> Pegasus vehicle in Isaac

bc_flight_supervisor + bc_episode_monitor
  -> lifecycle, termination, result JSON, traces, plots
```

`bc-eval` 是正式的 Isaac/Pegasus/PX4 飛行評估流程。

## 主要 ROS 2 nodes

| Node | 正式用途 | 職責 |
|---|---|---|
| `scene_bridge` | 資料收集、BC 評估 | Isaac pose/status -> 經驗證的場景障礙物、起點與終點 |
| `astar_planner` | 資料收集 | Scene -> A* 路徑 |
| `trajectory_parameterizer` | 資料收集 | Path -> timed trajectory |
| `trajectory_follower` | 資料收集 | Trajectory -> `ASTAR_EXPERT` command candidate |
| `expert_dataset_recorder` | 資料收集 | 將影像、odometry、expert command 和飛行證據同步記錄成 episodes |
| `px4_odometry_bridge` | 資料收集、BC 評估 | PX4 vehicle odometry -> ROS odometry |
| `control_mux` | 資料收集、BC 評估 | 唯一選取一個 command source |
| `px4_live_telemetry_adapter` | 資料收集、BC 評估 | 將 PX4 status/control/odometry/failsafe 轉為 gate 使用的 telemetry |
| `px4_mapping_gate` | 資料收集、BC 評估 | Fail-closed command 驗證與 PX4 candidate mapping |
| `px4_setpoint_streamer` | 資料收集、BC 評估 | 唯一向 PX4 發布 trajectory-setpoint/offboard-mode 的 node |
| `px4_vehicle_command_owner` | 資料收集、BC 評估 | 唯一向 PX4 發布 vehicle command 的 node |
| `px4_sitl_flight_supervisor`, `px4_sitl_flight_monitor` | 資料收集 | Expert 任務生命週期與飛行證據 |
| `bc_policy` | BC 評估 | TOP RGB AE+BC 推論 -> BC command candidate |
| `bc_flight_supervisor` | BC 評估 | Arm/takeoff/enable/landing 生命週期 |
| `bc_episode_monitor` | BC 評估 | 判定 goal/collision/timeout 並產生結果 artifact |

## Launch graph 對照

| Launch file | 使用者 | 該 graph 專屬的 nodes |
|---|---|---|
| `uav_px4_control/launch/px4_sitl_flight.launch.py` | `expert-collect` | A* planner、parameterizer、follower、expert recorder、expert supervisor/monitor |
| `uav_px4_control/launch/bc_flight.launch.py` | `bc-eval` | BC policy、BC supervisor、BC episode monitor；刻意不包含 planner/follower |

## 分類

### 主要元件

- `uav`
- `uav_ml/tools/{expert_collect,persistent_runtime,bc_baseline,bc_flight_evaluation,training_cli}.py`
- `uav_ml/train_autoencoder.py`, `uav_ml/train_bc.py`
- 正式 dataset、AE/BC model 與 BC inference modules
- `isaac/runtime/{bootstrap,environment,formal_expert_sensor_contract}.py`
- 上述來自 `uav_navigation`、`uav_scene_bridge`、`uav_data_recorder` 和 `uav_px4_control` 的
  active nodes 與 launch/config files

### 支援元件

- Expert 資料收集驗證、finalization 與 visual-QA helpers
- BC 飛行繪圖 helper
- 正式 navigation 與 PX4 設定 YAML files

### 診斷工具

- `scripts/diagnostics/summarize_bc_startup.py`
- Offline harnesses、比較工具、smoke checks 與 PX4 診斷 launch paths

### 不屬於正式流程

- `uav_camera_bridge` 和 `uav_bringup/uav_system_scaffold.launch.py` 是 scaffold nodes，不會由正式 graph 啟動。
