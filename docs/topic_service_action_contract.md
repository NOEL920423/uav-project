# 正式 ROS 2 通訊規約

本文件描述目前收集與 BC 飛行流程；詳細 schema 以 `ros2_ws/src/uav_interfaces/` 為準。
座標與時間規約見 [coordinate_frames.md](coordinate_frames.md)。

## 資料流與 ownership

| Topics／services | Owner／用途 |
|---|---|
| `/uav/isaac/runtime_status`、影像 streams | `isaac/runtime/environment.py` 的 embedded bridge |
| `/uav/isaac/scene/{obstacles,start,goal}` | `scene_bridge_node` 驗證 runtime、frame、stamp 後發布 |
| `/uav/scene/{obstacles,start,goal}` | BC launch 將 scene snapshots remap 到此處 |
| `/uav/planner/path` | `astar_planner_node`，僅 expert 收集啟動 |
| `/uav/trajectory/candidate`、`valid`、`status` | `trajectory_parameterizer_node` |
| `/uav/control/astar_command` | `trajectory_follower_node` 的 expert candidate |
| `/uav/vehicle/odometry` | `px4_odometry_bridge_node`，PX4 NED 狀態 |
| `/uav/bc/policy_status`、`/uav/bc/set_enabled` | `bc_policy_node`，含 readiness 原因及 action provenance |
| `/uav/control/selected_command`、`source`、`mux_status` | `control_mux_node` 是唯一 selected-command publisher |
| `/uav/control/set_source` | `control_mux_node`，來源切換經過 HOLD barrier |
| `/uav/px4/setpoint_candidate`、`output_gate_status`、`safe_to_forward` | `px4_mapping_gate_node`，驗證 coherent selected command／mux status 與 telemetry |
| `/uav/px4/set_output_enable` | Gate 明確啟用、停用與復原 |
| `/uav/px4/stream_status`、`set_stream_enable` | `px4_setpoint_streamer_node` |
| `/fmu/in/trajectory_setpoint`、`/fmu/in/offboard_control_mode` | 只有 streamer 可發布 |
| `/fmu/in/vehicle_command` | 只有 `px4_vehicle_command_owner_node` 可發布 |

## 必須維持的契約

- Scene 的 obstacles、start、goal 必須形成 coherent snapshot；空間訊息保留 frame。
- Expert／BC 先產生 candidate，由 mux 選取，再經 gate 與 streamer 發送至 PX4。
- Gate 使用相同 header stamp 配對 selected command 與 mux status；source 字串不參與拼接。
- 既有 freshness、速度／加速度限制、SITL identity、HOLD latch 與 explicit enable 行為維持。
- BC status 的 `ready=false` 代表沒有符合條件的可用動作；需同看 `reason` 才能判斷原因。
- Host publisher API 的完成時間不代表 PX4 已收到或執行；未校準時鐘時不得宣稱跨時鐘 latency。
- `Px4SyntheticTelemetry` 雖沿用舊名稱，也是 production live telemetry adapter 的資料介面。

純演算法、ROS contract 與控制 ownership 的 tests 保留於各套件 `test/`。
未實作的 episode action、placeholder camera/recorder 與 standalone offline graphs 已移除。
