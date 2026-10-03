# ROS 2 Python 維護複雜度稽核

**範圍：** `ros2_ws/src`；以 package metadata、`setup.py` entry points、launch files、topic/service contracts 及必要的 node source 為依據。盤點到 7 個 ROS 套件、141 個 Python 檔案。此文件是 inspection 報告；未修改 ROS 程式、launch、topic 或 runtime 行為。

## 1. Current architecture summary

ROS 2 workspace 以 ament Python packages 分開 scene bridge、navigation、data recording、PX4 control 和 bringup；`uav_interfaces` 是 ament CMake typed interfaces。主要程式分為 node adapters 與已有獨立模組的純演算法、狀態機、validator 和 diagnostics。

實際 flight launches 是多個互斥 mission graph，並非一條所有 node 都同時運作的固定 pipeline：

- **A* / SITL：** scene（外部場景可選）→ A* planner → trajectory parameterizer → trajectory follower → mux → PX4 mapping gate → setpoint streamer → PX4。
- **BC flight：** scene bridge + 外部影像來源 → BC policy → mux → mapping gate → streamer → PX4；此 launch 不啟動 A*、parameterizer 或 follower。
- **Offline：** 使用 synthetic candidate、kinematic plant、monitor 等 harness，不等於 Isaac/PX4 closed-loop flight。
- `uav_system_scaffold.launch.py` 是 Phase 1 idle placeholders；它與以上 flight launches 不同，並不代表它啟動了真正的 navigation、camera、PX4 control 或 recording capability。

目前 data/control path 的節點責任界線合理且應保留：scene bridge 驗證／轉接場景；planner、trajectory generator、follower 分階段處理規劃及控制 candidate；mux 唯一決定 command source；mapping gate 做安全檢查與 mapping；streamer 是 `/fmu/in/trajectory_setpoint`、`/fmu/in/offboard_control_mode` 的唯一 publisher；vehicle-command owner 和各 flight supervisor 負責另一組 PX4 lifecycle control。這些不是可互換的格式轉換 wrapper。

### ROS 2 runtime node 摘要

表內 topic 為主要介面（`…` 代表相關狀態／診斷 topic）；`—` 表示此 node 無該類 endpoint。launch 欄指 repo 內有明確啟動宣告，並不表示每次執行都會啟動（conditional launch 另註）。

| Package / file | 主要責任 | Publishers | Subscribers | Services / actions | UAV workflow launch | Merge candidate / risk |
|---|---|---|---|---|---|---|
| `uav_scene_bridge/scene_bridge_node.py` | 驗證 Isaac runtime/scene 狀態並發布場景快照 | `/uav/isaac/scene/{obstacles,start,goal}`, bridge status | `/isaac_uav/pose`, runtime status | — | BC、SITL（SITL conditional） | 不合併；與 planner 介面耦合、失敗須可觀測；中高風險 |
| `uav_navigation/planner/astar_planner_node.py` | 場景座標轉換、A*、簡化/B-spline candidate 與 path validation | planner raw/simplified/B-spline/final paths、validity/status | obstacles/start/goal 等一致快照 | — | SITL、offline | 不與 trajectory/follower 合併；高耦合但各階段可獨立驗證；高風險 |
| `uav_navigation/trajectory/trajectory_parameterizer_node.py` | 將 validated path 參數化為 timed trajectory | `/uav/trajectory/candidate`, `/valid`, `/status` | `/uav/planner/path` | — | SITL、offline | 保持獨立；中高風險 |
| `uav_navigation/tracking/trajectory_follower_node.py` | 追蹤 timed trajectory，輸出 A* candidate 和 tracking diagnostics | `/uav/control/astar_command`, reference/status | trajectory、validity、`/uav/vehicle/odometry` | `/uav/control/set_tracking_enable` | SITL、offline；BC 不啟動 | 不與 mux/gate 合併；candidate boundary 是隔離點；高風險 |
| `uav_px4_control/flight/bc_policy_node.py` | 影像/odometry/goal preprocessing 與 BC inference | `/uav/control/bc_command`, policy status | 選定 image topic、odometry、scene goal | `/uav/bc/set_enabled` | BC flight | 不與 follower 合併；不同 controller candidate，應由 mux 切換；中高風險 |
| `uav_px4_control/control/control_mux_node.py` | candidate freshness、source arbitration、HOLD handoff | `/uav/control/selected_command`, source、mux status | A*/BC/joystick/NavRL/HOLD candidates | `/uav/control/set_source` | BC、SITL、offline | **必須獨立**；單一 arbitration/ownership 與 HOLD 可觀測性；極高風險 |
| `uav_px4_control/px4/px4_odometry_bridge_node.py` | 驗證 PX4 NED odometry 並轉為 follower 使用的 ROS `Odometry` | `/uav/vehicle/odometry` | `/fmu/out/vehicle_odometry` | — | BC、SITL | 不和 telemetry adapter 合併；雖同讀 raw odometry，輸出轉換契約不同；高風險 |
| `uav_px4_control/px4/px4_live_telemetry_adapter.py` | 將 PX4 status/mode/odometry/failsafe 證據適配至 gate contract | `/uav/test/px4/telemetry_status` | 四個 `/fmu/out/*` safety telemetry topics | — | BC、SITL、PX4 stream | 不與 odometry bridge/streamer 合併；read-only safety evidence 要獨立可觀測；高風險 |
| `uav_px4_control/px4/px4_mapping_gate_node.py` | selected command mapping、候選驗證與 output gate | `/uav/px4/setpoint_candidate`, gate status、`safe_to_forward` | selected command、mux status、telemetry | `/uav/px4/set_output_enable` | BC、SITL、offline | **必須獨立於 mux/streamer**；安全 gate 與 publisher boundary 不應共用失效域；極高風險 |
| `uav_px4_control/px4/px4_setpoint_streamer_node.py` | 經獨立 readiness/safety gate 後輸出 PX4 setpoints | 唯一發布 `/fmu/in/trajectory_setpoint`、`/fmu/in/offboard_control_mode`；stream status | candidate、gate/safe 狀態、PX4 status/mode/odometry/failsafe | `/uav/px4/set_stream_enable` | BC、SITL、stream diagnostic | **必須獨立**；PX4 實際輸出邊界；極高風險 |
| `uav_px4_control/flight/px4_vehicle_command_owner_node.py` | 唯一 PX4 vehicle-command publisher | `/fmu/in/vehicle_command` | — | `/uav/px4/send_vehicle_command` | BC、SITL flight | **必須獨立於 streamer**；command lifecycle 與 setpoint stream 是不同權限；極高風險 |
| `uav_px4_control/flight/bc_flight_supervisor_node.py` | BC mission orchestration / readiness / recovery | BC flight status、lifecycle | telemetry、bridge、episode、vehicle state 等 | 呼叫 mux/gate/stream/policy/vehicle-command services | BC flight | 不與 SITL supervisor 合併；任務狀態機不同；高風險 |
| `uav_px4_control/flight/px4_sitl_flight_supervisor_node.py` | A* SITL mission orchestration | flight/goal/status | scene/planner/tracking/telemetry/vehicle evidence | 啟動 tracking/mux/gate/stream/vehicle-command clients；提供 flight control service | SITL flight | 不與 BC supervisor 合併；各自保持獨立 mission lifecycle；高風險 |
| `uav_px4_control/flight/bc_episode_monitor_node.py`、`diagnostics/px4_sitl_flight_monitor.py` | 監測／記錄 flight 結果，非控制器 | episode termination 或 evidence | scene、odometry、policy/supervisor/stream 狀態等 | —（SITL monitor 可呼叫 supervisor） | BC、SITL | 保持只讀觀測面；合併會擴大控制與監測失效域；中高風險 |
| `uav_data_recorder/expert_dataset_recorder_node.py` | 記錄 expert episode/dataset | —（輸出 dataset artifacts） | scene、vehicle、影像/trajectory 等資料 | — | SITL（由 launch argument 控制） | 不併入控制 nodes；資料同步失敗須獨立診斷；中風險 |
| `uav_data_recorder/episode_scene_client.py` | 發出 seeded scene prepare request 前確認 PX4 landed/disarmed | `/uav/isaac/episode_command` | Isaac runtime status、PX4 vehicle/land status | — | repo scripts/runtime tooling；不在 flight launch | 保持獨立的安全 preflight CLI；並非 orphan；中風險 |
| `uav_camera_bridge/camera_bridge_node.py`、`uav_navigation/navigation_node.py`、`uav_px4_control/px4_control_node.py`、`uav_data_recorder/data_recorder_node.py` | Phase 1 空殼 node；宣告參數後 spin，不建立資料/控制 endpoints | — | — | — | 僅 `uav_system_scaffold.launch.py` | 僅重複 idle-node 樣板；不屬 flight graph。合併會改變 node identity/launch graph，無足夠收益；低至中風險但不建議 |

`offline_tracking_harness.py`、`offline_control_mux_harness.py`、`offline_px4_boundary_harness.py` 與 planner/parameterizer 檔案中的 Offline harness classes 也含 ROS nodes；它們是測試/diagnostic publishers、plant、monitor，不是 production control nodes。它們依 offline launch 使用，不能當未使用程式刪除。

## Python 檔案分類覆蓋表

下列路徑規則對 `ros2_ws/src` 盤點出的 141 個 `.py` 檔案完整分類；「除明列例外」表示其餘該目錄檔案全部歸入該類，沒有證據足夠的 `Unknown` 或 legacy implementation 不會硬標為已棄用。

| 分類 | 檔案範圍 |
|---|---|
| **1. ROS2 runtime node** | 上表列出的 nodes；另有 utility/offline nodes：`uav_data_recorder/episode_scene_client.py`、`uav_navigation/diagnostics/offline_tracking_harness.py`、`uav_px4_control/diagnostics/{offline_control_mux_harness.py,offline_px4_boundary_harness.py,px4_generation_probe.py,px4_sitl_doctor.py,px4_sitl_stream_monitor.py,px4_sitl_flight_monitor.py,runtime_smoke_lifecycle_client.py}`。`astar_planner_node.py` 和 `trajectory_parameterizer_node.py` 亦定義 Offline harness nodes。 |
| **2. CLI / executable entry point** | 所有 `*/setup.py`（package/console-script metadata）；不以 ROS Node 為主的 CLI modules：`uav_navigation/diagnostics/{geometric_comparison.py,tracking_comparison.py}`、`uav_px4_control/diagnostics/{control_mux_comparison.py,px4_stream_fixtures.py}`。其餘 `setup.py` 指向的 ROS Node modules 均歸類為 runtime node（類別 1），即使它們同時提供 console script。 |
| **3. Shared library / helper** | 所有 package `__init__.py`；`uav_data_recorder/expert_dataset_contract.py`；`uav_scene_bridge/runtime_contract.py`；`uav_navigation/planner/` 除 `astar_planner_node.py` 外的 Python modules、`tracking/{tracking_models.py,tracking_validator.py,trajectory_tracker.py}`、`trajectory/{trajectory_metrics.py,trajectory_models.py,trajectory_parameterizer.py,trajectory_sampler.py,trajectory_validator.py,yaw_profile.py}`、`diagnostics/{offline_kinematic_plant.py,tracking_fixtures.py,tracking_metrics.py}`；`uav_px4_control/control/` 除 `control_mux_node.py` 外的 modules、`flight/{bc_episode_monitor.py,bc_flight_models.py,px4_flight_models.py,px4_flight_state_machine.py}`、`px4/` 除 `*_node.py` 和 `px4_live_telemetry_adapter.py` 外的 model/validator/mapper/state/timestamp modules、`diagnostics/{control_mux_fixtures.py,px4_synthetic_telemetry.py}`。 |
| **4. Launch / config-related helper** | `uav_bringup/launch/uav_system_scaffold.launch.py`、`uav_navigation/launch/*.launch.py`、`uav_px4_control/launch/*.launch.py`（合計 14 個 launch Python files）。Offline launch 使用的 harness modules 依其主要 node 職責仍歸類為類別 1。 |
| **5. Test** | 所有 `*/test/*.py`（含 flake8/pep257、contract、unit、ROS integration tests）。 |
| **6. Legacy / 可能未使用** | **沒有確認為 legacy 或無引用的 executable/module。** Phase 1 placeholders（camera、navigation、PX4 control、data recorder）只供 scaffold launch，屬未實作的預留邊界，不等於已廢棄 legacy。 |
| **7. Unknown** | 目前無法判定項目：無。對動態 import、外部 launch/ROS CLI 使用及未在 repo 內的 Isaac image publishers，文字搜尋不能證明全域未使用。 |

## 2. Files safe to investigate for deletion

沒有檔案能僅憑 repo 內證據判定為可直接刪除。優先確認 scaffold 是否仍需維護，再評估：

1. `uav_camera_bridge/uav_camera_bridge/camera_bridge_node.py`：目前只宣告 disabled placeholder，沒有建立 image/camera-info publisher；只由 Phase 1 scaffold launch 啟動。真實 flight graph 沒有啟動它，但實際影像來源可能在 repo 外，先確認外部依賴。
2. `uav_navigation/uav_navigation/navigation_node.py`、`uav_px4_control/uav_px4_control/px4_control_node.py`、`uav_data_recorder/uav_data_recorder/data_recorder_node.py`：同樣是 scaffold-only idle wrappers，且各自被 `uav_system_scaffold.launch.py` 使用。只有停止支援該 scaffold 時，才進一步考慮移除其 launch/entrypoint/placeholder。
3. `uav_bringup/launch/uav_system_scaffold.launch.py`：只啟動五個 disabled placeholders；若團隊確認已不再需要 Phase 1 scaffold，可與上述 wrappers 一起做依賴清理評估。

**不列入刪除候選：** offline harness、test/diagnostic executables、`episode_scene_client.py`（有 shell/runtime callers）、expert recorder、shared contract modules。現有 setup entry points 均能找到 launch、測試、文件或 repo script 的引用；未發現可確認的 orphan executable。可能超出 repo 的呼叫仍需部署端確認。

## 3. Low-risk consolidation candidates

- **不建議合併任何 production ROS node。** 目前沒有既能降低實質維護複雜度、又能維持隔離與 observability 的低風險 node merge。
- 若後續確有維護痛點，可先比較 `bc_flight.launch.py` 與 `px4_sitl_flight.launch.py` 共用 PX4 node/config wiring；僅在 launch 展開後 node 數、名稱、參數、remappings、conditions 完全不變時，才考慮抽共用 launch helper。先做 graph equivalence 測試，不要合併控制節點。
- Idle scaffold wrappers 雖重複 lifecycle 樣板，但 node names、parameters 和 launch 可見性不同；抽象成 generic placeholder 會新增間接層，預期收益低，不列為優先整合。

## 4. Nodes that should remain separate

- **Mux、mapping gate、setpoint streamer、vehicle command owner：** 分別負責 source arbitration、輸出驗證、安全使能、PX4 setpoint stream、vehicle lifecycle command。Topic ownership tests 和獨立 gate/status 是安全邊界，不可為減少 node 數而合併。
- **Telemetry adapter 與 odometry bridge：** 都訂閱 `/fmu/out/vehicle_odometry`，但前者產生 gate 用 safety evidence，後者驗證 NED frame/quaternion 並輸出 `/uav/vehicle/odometry` 給 controller。共享 raw input 不代表責任重疊。
- **A* planner、trajectory parameterizer、follower：** path validation、時間參數化及 feedback control 的資料契約和失敗條件不同；分離有利於逐段 HOLD 與離線驗證。
- **BC policy 與 A* follower：** 是互斥的 controller candidates，應由 mux 選擇；不要把 controller 和 mux 合併。
- **BC/SITL supervisors、flight monitors、recorders：** mission-specific orchestration、只讀監測與資料保存應維持獨立，避免控制與證據收集共享故障。

### Topic / wrapper / overlap observations

- `/uav/isaac/scene/{obstacles,start,goal}` 與 `/uav/scene/{obstacles,start,goal}` 由 BC launch remapping 對接；是 namespace alias，不是兩份相同資料 producer。
- A*/BC candidate、mux selected command、PX4 candidate、PX4 `/fmu/in/*` 是分層 endpoints，刻意分開，不是應去重的重複 topics。
- `px4_live_telemetry_adapter` 把**真實 PX4 live telemetry**適配到既有 `/uav/test/px4/telemetry_status` contract；`/test/` 名稱容易誤讀，但目前不要改名，以免破壞 gate/streamer contract。
- 很小的 wrappers 是四個 Phase 1 placeholder nodes。它們只 spin，無 topic/service；但仍是 scaffold graph 的獨立 node identity，不能視為無引用或無行為影響。
- BC policy 訂閱 `/uav/isaac/observer/image/compressed`、`/uav/isaac/fpv/image/compressed` 或 depth topic；目前 workspace 的 `camera_bridge_node.py` 沒有產生這些資料，且 BC launch 不啟動 camera bridge。影像來源是外部 dependency/待補 boundary，單靠本次 `ros2_ws/src` 稽核無法定位。

## 5. Recommended refactoring order

1. 保留本 inventory，先由部署/實驗流程確認 Phase 1 scaffold、外部 camera/Isaac publishers、所有 `ros2 run` scripts 是否仍在使用。
2. 若 scaffold 已退役，先獨立移除 scaffold launch/entrypoint 的計畫並檢查外部呼叫；不要和 production flight nodes 一起改。
3. 對 offline executable 做明確 owner/reference 清單；只處理完成 repo 外 caller 確認的 orphan，不憑未出現在某個 flight launch 就刪除。
4. 如 launch wiring 重複造成真實維護問題，先在 BC/SITL launch 層小幅整理並比對 launch graph；不更動 topic、params、conditions 或 node identity。
5. 最後才評估 production node 邊界；若真的需要調整，先保留 mux/gate/streamer/vehicle-command ownership，跑相應 contract/unit tests，再做 targeted SITL 驗證。不要以 Python 檔案數量當重構目標。
