# ROS 2 A* / B-spline 整合審計

審計日期：2026-08-05（Asia/Taipei）

審計範圍：`~/uav-project`、`~/uav_ros2_ws`，以及相鄰的 Isaac Sim、PX4、Pegasus、Micro XRCE-DDS checkout。

狀態：Phase 0 靜態審計完成；未啟動 Isaac Sim、PX4，未 arm、未 flight。

## 1. 結論摘要

目前不是「完全非 ROS 2」的專案，而是兩條並存的實作：

1. 根目錄 legacy pipeline：場景、相機、PNG recorder、A*、PX4 控制都在 Isaac Sim Python process 內，以 `builtins` 串接，`4.px4_astar.py` 直接使用 `pymavlink`。
2. 半 ROS 2 pipeline：Isaac 端仍用 `builtins` 執行場景、相機、recorder 與 A*，只把 `nav_msgs/Path` 與 Isaac pose 發到 ROS 2；外部 `uav_px4_control` 已用 `px4_msgs` 做 path following、OFFBOARD、ARM、LAND 與 mission orchestration。

可保留的工作成果很多：現有 A* 有 obstacle inflation、RDP/greedy fallback、連續 segment validation、2.5D short-obstacle overflight、lookahead 與 lookahead shortcut validation；ROS 2 follower 已有 telemetry freshness、failsafe、command ACK、OFFBOARD/ARM confirmation 與 landing recovery。遷移應抽取並測試這些行為，不應重寫後丟失。

目前缺口是：沒有 obstacle/start/goal ROS contract、沒有 camera image topics、沒有 B-spline、沒有 controller multiplexer、沒有統一 EpisodeState/custom action、沒有 rosbag/同步 imitation-learning manifest，也沒有 Phase 0 規格要求的單元或非飛行 integration tests。

## 2. 儲存庫與工作區基線

審計開始時，`/home/noel_614420090/uav-project` 沒有 `.git`，所以原始 `git status`、branch 與 log 都回覆 `fatal: not a git repository`。為滿足可回復與分 phase commit 要求，已採取下列非破壞性處理：

- 在 `~/uav-project` 初始化 `main`。
- 既有 4.5 GB generated datasets/logs、模型與本地 vendor copy 留在原位，但由 `.gitignore` 排除。
- 建立 baseline commit `e198a71`（`chore: preserve pre-migration baseline`）。
- 建立 branch `feature/ros2-astar-bspline-integration`。

`~/uav_ros2_ws` 本身也不是 Git repository；只有 `src/px4_msgs` 是獨立 repository。這是後續 commit 邊界的重大風險。Phase 1 起應把 canonical ROS 2 source 放入本 repository 的 `ros2_ws/src`，以外部 `~/uav_ros2_ws` 作 runtime build/install workspace；舊 workspace source 在等價驗證前保持不動。

### 已執行的核心基線命令

```bash
pwd
git status
git branch --show-current
git log -5 --oneline
printenv ROS_DISTRO
lsb_release -a
python3 --version
python --version
find .. -maxdepth 4 -name .git -type d -print
ls -la /opt/ros
cat ../isaacsim/VERSION
git -C ../PX4-Autopilot describe --always --tags --dirty
git -C ../PegasusSimulator describe --always --tags --dirty
git -C ../Micro-XRCE-DDS-Agent describe --always --tags --dirty
source /opt/ros/jazzy/setup.bash
source ../uav_ros2_ws/install/setup.bash
ros2 pkg list
timeout 15s ros2 topic list -t
timeout 15s ros2 service list -t
ros2 node list
```

所有命令都在 tmux session `astar` 執行；審計期間沒有使用 `sudo`、安裝套件或修改網路設定。

## 3. 環境版本

| 項目 | 實際審計結果 | 備註 |
|---|---|---|
| Ubuntu | 24.04.4 LTS (noble) | `lsb_release -a` |
| ROS | Jazzy | shell 初始未 source；source 後 `ROS_DISTRO=jazzy`；另裝有 Rolling |
| default `python` / `python3` | 3.10.13 | pyenv 路徑 |
| `/usr/bin/python3` | 3.12.3 | ROS packages 使用的 system Python |
| Isaac embedded Python | 3.11.13 | `~/isaacsim/_build/linux-x86_64/release/python.sh --version` |
| Isaac Sim | `5.1.0-rc.19` | `~/isaacsim/VERSION`；實際 runtime 在 `_build/linux-x86_64/release` |
| Pegasus Simulator | `v5.1.0` | `~/PegasusSimulator` clean tag |
| PX4 | `v1.14.3-dirty` | detached HEAD；有已存在修改與 untracked runtime files |
| `px4_msgs` | `v1.14.0` | clean `release/1.14` branch |
| Micro XRCE-DDS Agent | `v2.4.3` | checkout tag |
| `uav_px4_control` | `0.5.0` | `setup.py` / `package.xml` |

PX4 的既有 dirty state：

```text
M Tools/setup/requirements.txt
M src/modules/uxrce_dds_client/dds_topics.yaml
?? Tools/setup/requirements.txt.bak
?? dataman
?? etc
```

這些不是本次變更，後續不得覆蓋或清除。`dds_topics.yaml` 已修改，表示 ROS topic surface 不可依上游預設猜測，integration 前必須在 PX4 運行時重新 `ros2 topic list -t`。

### Python dependency 審計

| 環境 | NumPy | SciPy | pymavlink | rclpy | cv_bridge |
|---|---:|---:|---:|---:|---:|
| default Python 3.10.13 | 2.2.6 | 未安裝 | 2.4.49 | 7.1.11 | 4.1.0 |
| `/usr/bin/python3` / ROS | 1.26.4 | 未安裝 | 未檢出 | 7.1.11 | 4.1.0 |

因此 B-spline 不得直接加入 `scipy.interpolate` 依賴。優先設計純 NumPy de Boor / clamped B-spline；若後續選 SciPy，必須先說明只安裝到 ROS workspace 可控環境，不能修改 Isaac Sim embedded Python。

## 4. ROS 2 workspace 與執行中 graph

現有 workspace：`/home/noel_614420090/uav_ros2_ws`。

已建置、與本專案直接相關的 packages：

- `px4_msgs`
- `uav_px4_control`
- 系統另有 `cv_bridge`、完整 `rosbag2` storage/transport packages

沒有找到：`uav_interfaces`、`uav_scene_bridge`、`uav_camera_bridge`、`uav_navigation`、`uav_data_recorder`、`uav_bringup`。

審計時 Isaac Kit、PX4 SITL 與 Micro XRCE-DDS Agent 都沒有運行。存在一個 BC inference worker 與 ROS daemon，但不是飛行 stack。實際 ROS graph：

```text
Topics:
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/rosout [rcl_interfaces/msg/Log]

Services: none
Nodes: none
```

所以本次只確認了 installed message/package surface，沒有聲稱 live PX4 topics 或飛行整合成功。

## 5. 主要檔案審計

### 5.1 `1.dual_uav_camera.py`

- 與 `ros2_isaac_scripts/1.dual_uav_camera.py` SHA-256 完全相同。
- 建立 `/World/UAV_Camera_FPV` 與 `/World/UAV_Camera_Observer`，追蹤 `/World/quadrotor/body`。
- 使用 `UsdGeom.Camera.Define`、`Gf.Matrix4d().SetLookAt()` 與 Kit update event subscription 更新 camera transform。
- FPV 可依 body axis 或 motion direction；observer 支援 TOP/CHASE。
- 使用 `omni.kit.viewport.utility` 自動分配/建立 viewport，但這只負責 GUI 顯示。
- lifecycle 與 runtime controls 都掛在 `builtins`。
- 不發 `sensor_msgs/Image`、`CameraInfo` 或 TF。

Isaac Sim 5.1 適配風險：viewport APIs 以 optional import 包裝；建立 camera 的 USD API穩定，但不能把 viewport 當正式 image source。

### 5.2 場景產生器

根目錄 `2.scene_episode_generator.py` 是較舊 cylinder 版本；ROS pipeline 使用 `ros2_isaac_scripts/2.scene_episode_generator.py` 的新版本：

- 唯一 generated root：`/World/GeneratedEpisode`。
- start `(0,0,0)`，target `(3,5,0)`；預設 8 obstacles。
- 新版建立可碰撞 high-rise building，planning radius 是旋轉前 width/depth half-diagonal。
- 新版固定插入兩個 direct-path blockers，且加入 episode lighting。
- obstacle、start、target 會以 custom USD metadata 寫入 live stage。
- 可設 `RANDOM_SEED`，但現有 `std_srvs/Trigger` service 無法在 request 傳 seed/count。
- JSON 包含 episode ID、timestamp、seed、root、start/target 與 object records；CSV 包含 shape/pose/radius/width/depth/height/yaw/collision/color/placement mode。
- scene regeneration 只有 manager 的 process-local `busy` guard，沒有 armed/episode-active interlock。

### 5.3 PNG recorder

需求中提到的 `3.front_camera_png_recorder.py` 不存在。實際檔名是 `3.dual_camera_png_recorder.py`；`run_front_camera_recorder.py` 仍指向不存在的舊檔名，是已確認的 broken legacy wrapper。

根目錄 recorder 仍是 viewport capture；ROS pipeline 的 `ros2_isaac_scripts/3.dual_camera_png_recorder.py` 已改善為：

- Isaac Replicator off-screen render products，不依賴 active viewport。
- 每個 camera 配一個 RGB annotator。
- 以 simulation time 排程，預設 10 FPS；sim time 不可用時才 fallback wall time。
- 成對取得 FPV/TOP arrays，兩張都成功才落盤；失敗會移除半套輸出。
- 每 episode 產生 `images/fpv`、`images/top` 與 `camera_frames.csv`。
- CSV 已記 episode ID、frame index、wall/record/sim time、兩張 image path、camera/body prim、Isaac pose/yaw、clock source 與 resolution。
- start/stop 仍透過 legacy `builtins.start_front_camera_png_recorder()`。
- 尚未發 ROS image topics、CameraInfo，也沒有 rosbag integration。

### 5.4 `4.px4_astar.py`

既有 working safety/control 行為：

- 從 live USD stage 讀 top-level obstacles；以 bbox/custom metadata估計 center、radius、height。
- 2D 8-connected A*，resolution 0.05 m，grid margin 2.0 m。
- clearance-aware soft cost、direct-path bias 與 occupied endpoint recovery。
- A* planning forbidden radius：`obstacle_radius + 0.18 + 0.13`。
- continuous segment validation 再加 0.07 m；即驗證 threshold 為 `obstacle_radius + 0.38 m`。
- 先 RDP simplify + densify；失敗後 greedy safe simplify；再失敗回 raw dense A*；仍 unsafe 則 planner abort，且在 connect PX4 前停止。
- `GRID_DISCRETIZATION_MARGIN_M ~= 0.035 m` 出現在 diagnostic effective radius，但 backward-compatible actual occupancy radius 沒有使用它；continuous validation 可攔截 path segment，但這個公式分歧必須在抽取時由 regression tests 固定或修正。
- short obstacle 若 `top_z + 0.35 <= 2.0 m`，可從 2.5D planning obstacle set 排除。
- lookahead 會從 0.55 m 起縮短 carrot，並針對「目前 UAV 到 carrot」額外做 0.07 m continuous segment check。
- direct MAVLink：`udpin:0.0.0.0:14550`；20 Hz velocity+yaw-rate `SET_POSITION_TARGET_LOCAL_NED`；OFFBOARD、ARM、ACK/heartbeat confirmation、takeoff、mission、auto-land。
- command acceleration limiter、dynamic speed scaling、keyboard land。
- mission CSV 含 state、pose/velocity/attitude、command、goal、nearest obstacle、clearance、path progress、planner metadata。
- camera recording start/stop 仍透過 `builtins`。

已知缺口：沒有 collision sensor state、沒有 odometry timestamp freshness check、沒有 command publisher exclusivity、state 名稱不符合目標 EpisodeState、metrics 沒有 curvature/jerk/RMSE，且所有責任在單一 4,124 行檔案。

## 6. 現有 ROS 2 實作

### Isaac-side nodes

`ros2_isaac_scripts/5.astar_ros2_path_publisher.py`：

- 約 4,533 行，是 A* runner 的大幅複製，而非可 import planner module。
- `PUBLISH_ROS2_PATH_ONLY=True`，發 `/uav/planned_path` (`nav_msgs/Path`)。
- `frame_id=px4_ned`，1 Hz，reliable + transient local。
- 只發 final simplified path；沒有 raw/simplified/candidate/final 四層 topics。

`5.ros2_uav_pose_publisher_logger.py`：

- 發 `/isaac_uav/pose` (`geometry_msgs/PoseStamped`)，10 Hz，frame `isaac_world`。
- stamp 取 sim time；CSV 同時保留 wall/record/sim/ROS stamp。
- pose 是原始 Isaac XYZ/quaternion，沒有轉成 NED，也沒有 velocity。

`6.isaac_ros2_episode_manager.py`：

- 發 `/uav_sim/status`、`/uav_sim/episode_id` (`std_msgs/String`, transient local)。
- 提供 `/uav_sim/prepare_episode`、`cleanup`、`generate_scene`、`setup_cameras`、`start_recording`、`stop_recording`、`start_pose_logger`、`stop_pose_logger`、`plan_path`、`stop_all`、`get_status`，全部是 `std_srvs/Trigger`。
- manager 只是 ROS service 到 `runpy`/`builtins` 的 adapter。
- `prepare_episode` 依序 stop components、remove generated prims、generate scene、setup cameras、optional record/pose、plan path。

### PX4-side package

`uav_px4_control/lookahead_follower.py`：

- subscribe `/uav/planned_path`。
- publish `/fmu/in/offboard_control_mode`、`/fmu/in/trajectory_setpoint`、`/fmu/in/vehicle_command`。
- subscribe `/fmu/out/vehicle_local_position`、`vehicle_odometry`、`vehicle_status`、`vehicle_command_ack`、`vehicle_land_detected`。
- services `/uav_control/start_mission`、`reset_mission`、`abort_mission`、`get_status`。
- state topic `/uav_control/mission_state`。
- 預設 `auto_start_on_path=false`，收到 path 不會自行 arm。
- 已檢查 path frame/finite values、telemetry freshness、PX4 failsafe、OFFBOARD/ARM confirmation、unexpected disarm/mode loss 與 landing confirmation。

`mission_orchestrator.py`：

- services `/uav_mission/run_episode`、`abort`、`get_status`。
- topic `/uav_mission/state`。
- 將 prepare → follower start → ACTIVE → capture → landing/cleanup 串起來。
- 使用 `std_srvs/Trigger`，無 goal parameters、feedback、typed result 或完整 metrics。

`joystick_teleop.py`：

- 已有 deadman、stale joystick 零速度、arm/land/disarm buttons。
- 它直接發同一組 `/fmu/in/*` topics；若和 follower 同時 launch，沒有 mux 阻止雙 publisher。

未找到名稱完全相符的 `ros2_lookahead_follower_checked.py`；其 safety-check 行為已合併到現有 `lookahead_follower.py`。也未找到獨立的 ROS camera publisher。

## 7. 座標與時間契約

目前 planner 的 `SWAP_XY=True` 且 offsets 全為 0：

```text
Isaac [x, y, z] -> PX4 NED [y, x, -z]
PX4 NED [n, e, d] -> Isaac [e, n, -d]
ground start/goal 的 NED z 強制設為 -2.0 m
```

這不是一般 ENU→NED 的完整姿態轉換，只是此 scene/Pegasus local frame 的位置 mapping。Pose publisher 仍發 `isaac_world` 原始 pose；path follower 則要求 `px4_ned`。目前沒有 TF tree 或一個 canonical transform module，容易讓 camera/pose/dataset 混用 frame。

時間也不統一：

- path header 使用 ROS node clock；未明確設定 `use_sim_time`。
- Isaac pose/header 與 recorder scheduler 優先使用 Isaac simulation time。
- PX4 control freshness 使用 wall time。
- legacy mission log 使用 wall time。

目標 recorder 必須把 clock source 明確化，且保存 sim time 與 monotonic/wall diagnostic time，不能把它們視為同一時間基準。

## 8. 重複、過期與 broken implementations

- camera root 與 ROS copy相同，可暫時視為同一來源的 duplicate。
- scene root 是舊 cylinder；ROS copy 是新 high-rise/direct-blocker 版本。
- recorder root 是舊 viewport；ROS copy是 Replicator dual-camera 版本；另有 `_b` backup。
- pose logger root 與 ROS copy不同，ROS copy支援 episode ID/sim time。
- `4.px4_astar.py` 與 `5.astar_ros2_path_publisher.py` 重複 planner/safety code，已開始 drift。
- `5.astar_waypoint_exporter.py` 是另一份大型 planner/exporter。
- `~/uav_ros2_ws` 根目錄還留有 standalone `ros2_lookahead_follower.py`、`ros2_joystick_teleop_px4.py` 等，package 內也有相同功能的新版。
- 多個 `uav_ros2_phase2`、`uav_ros2_phase3`、`codex_backups`、`uav_ros2_backups` 是歷史副本，不應當作 canonical source。
- `run_front_camera_recorder.py` 指向不存在檔案。
- `uav_pipeline.sh` 綁定既有 `tmux uav` pane layout，並非通用 launch entry point。

Phase 0 不搬動或刪除任何 legacy file。

## 9. 主要遷移風險與 gate

| 風險 | 影響 | Phase gate / 緩解 |
|---|---|---|
| project 與 ROS workspace 分離且後者未版控 | 無法形成可審查 atomic commits | canonical source 放 `uav-project/ros2_ws/src`；舊 workspace先保留 |
| PX4 1.14 / px4_msgs 1.14 且 dds_topics 已改 | topic/field 不可照新文件猜測 | live integration 前 `ros2 interface show` + live graph |
| SciPy 不存在 | 直接引入會破壞 Isaac/ROS env | 優先純 NumPy B-spline；dependency decision另立 commit |
| planner 三份以上且已 drift | 修一份、另一份仍飛舊邏輯 | 先抽 pure modules + golden regression tests |
| actual/diagnostic inflation formula 不一致 | clearance statement可能誤導 | 用同一 `SafetyEnvelope` 資料型別與 tests 統一 |
| obstacle 用 bounding circle | 對矩形安全但保守，窄通道可能被封 | 首個 milestone 保持此行為，不先改幾何模型 |
| camera API混用 viewport/Replicator | capture 不穩或無 ROS image | 5.1 先延用已工作的 Replicator render product，再橋接 ROS |
| `builtins` 隱式 ownership | 重跑/cleanup/race 難控制 | service/action lifecycle逐元件替代；legacy fallback保留 |
| joystick/follower都直發 `/fmu/in/*` | 多 controller 同時控制 | controller mux 必須在任何新增自動飛行前完成 |
| 沒有 live stack | 無法證明 flight acceptance | 先 pure/non-flight tests；待使用者允許 auto-arm才整合 |
| dataset sync只有相同 episode/sim time | 沒有 ROS ApproximateTime manifest | recorder package建立統一 timestamp/episode contract |

## 10. Phase 0 architecture decisions

1. 保留 `ros2_isaac_scripts` 與 direct-`pymavlink` runner，直到 ROS 2 controller 完成 integration flight。
2. 新 ROS canonical source 放在本 repository 的 `ros2_ws/src`，避免繼續在未版控的 `~/uav_ros2_ws` 修改。
3. planner/smoothing/validation/metrics 先做純 Python modules，不 import Isaac、rclpy、pymavlink。
4. obstacle safety envelope 由單一 module與 immutable config產生，A*、simplifier、B-spline validator、lookahead validator共用。
5. B-spline candidate 永遠是 optional；validation failure 必須 publish false/reason並 fallback到已驗證 simplified/raw A*。
6. Isaac 5.1 camera 先沿用 off-screen Replicator render product，ROS bridge只包最小轉換；不回退 active viewport。
7. 外部 PX4 control 保留既有 `px4_msgs` 1.14 topic/field契約，先加 mux再接 expert/joystick/NavRL。
8. Episode orchestration 改為 typed action，但 emergency stop/health query可保留小型 service。
9. 在 non-flight tests通過前不啟動 PX4/Isaac integration；在 `auto_arm` 明確為 true 前不 arm。

## 11. Phase 0 未執行事項

- 沒有 build、unit test或 launch test，因尚未新增 code/package。
- 沒有啟動 Isaac Sim、Pegasus、PX4 或 Micro XRCE-DDS Agent。
- 沒有 flight、arm、land實測。
- 沒有安裝 SciPy或其他 package。
- 沒有宣稱 acceptance criteria 1–12 已完成。

下一階段應先建立 `uav_interfaces` 與 pure `uav_navigation` skeleton/tests；不應先碰 auto-arm 或 dataset bulk operations。
