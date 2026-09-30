# Topic、service、action 與 ownership 規約

## 範圍與慣例

本規約最初是 Phase 1 的目標。Phase 3 目前只實作 planning section 中的三個 scene
subscriptions 和六個非控制用 planning publications。其他 endpoints 仍屬於未來規約。

表格使用以下 QoS 簡寫：

- `R/TL/K1`：reliable、transient-local、keep-last depth 1。
- `R/V/K5`：reliable、volatile、keep-last depth 5。
- `BE/V/K5`：best-effort、volatile、keep-last depth 5（sensor-data style）。
- Services/actions 使用預設的 reliable service/action profiles；action feedback 為 volatile。

所有帶 stamp 的 interfaces 都遵循 `coordinate_frames.md` 中的 ROS clock 規約。在 simulation
中，設定 `use_sim_time=true` 並共用同一個權威 `/clock`。空間訊息的 frame 不得為空。

## Phase 8 僅限 SITL 的 PX4 streaming 規約

Phase 8 新增一個需另外啟用的 output boundary。唯一 publisher owner 為
`uav_px4_control.px4_setpoint_streamer_node`。Live allowlist 僅包含
`/fmu/in/trajectory_setpoint` (`px4_msgs/msg/TrajectorySetpoint`) and
`/fmu/in/offboard_control_mode` (`px4_msgs/msg/OffboardControlMode`)。Phase 8 nodes 不得發布
其他 `/fmu/in/*` topics。尤其禁止 `/fmu/in/vehicle_command`、OFFBOARD 請求、arm/disarm、
起飛和降落。

| 精確名稱與型別 | 擁有者 / 使用端 | 規約 |
|---|---|---|
| `/uav/px4/set_stream_enable` — `uav_interfaces/srv/SetPx4StreamEnable` | Server: sole streamer; client: explicit SITL test/operator | Enables only message streaming after all independent readiness gates; disable also clears a latched stream fault. It never changes vehicle mode or arming state. |
| `/uav/px4/stream_status` — `uav_interfaces/msg/Px4StreamStatus` | Streamer / monitor and operator | Reliable volatile status at stream timer rate; reports typed state, both enables, SITL/DDS/gate/candidate/telemetry evidence, vehicle safety state, timing, counts, and stop reason. |
| `/fmu/in/trajectory_setpoint` — `px4_msgs/msg/TrajectorySetpoint` | Sole streamer / PX4 SITL uXRCE subscriber | 20 Hz only while both gates and all live safety evidence pass. NED velocity identity mapping; unused fields and absolute yaw are NaN. Publication stops completely when disabled or faulted. |
| `/fmu/in/offboard_control_mode` — `px4_msgs/msg/OffboardControlMode` | Sole streamer / PX4 SITL uXRCE subscriber | Published as a pair with each trajectory setpoint; velocity flag only. This is a stream heartbeat, not a mode request. Publication stops completely when disabled or faulted. |

Streamer 也會讀取 `/fmu/out/vehicle_status`、
`/fmu/out/vehicle_control_mode`、`/fmu/out/vehicle_odometry` 和
`/fmu/out/failsafe_flags`。啟用前必須確認載具未 arm、OFFBOARD 未啟用、沒有 failsafe、NED
odometry 足夠新、DDS endpoints 正常，且具有本機 SITL 身分。

## Scene 規約

| 精確名稱與型別 | 擁有者 | 訂閱端 / clients | Frame 與 timestamp | QoS / 頻率 | 狀態限制 | 失敗行為 | 未來階段 |
|---|---|---|---|---|---|---|---|
| `/uav/scene/obstacles` — `uav_interfaces/msg/ObstacleArray` | `uav_scene_bridge` scene publisher | `uav_navigation` planner/validator、visualization、recorder | `header.frame_id=isaac_world`；stamp 為 ROS clock 記錄的已提交場景快照時間 | `R/TL/K1`；每次成功產生／更新時發布一次 | 只有 episode state 為 `READY` 且快照完整、通過驗證時才發布；所有尺寸必須是有限且非負值 | 不得以部分或無效資料取代最後一份有效快照；generation response/state 必須回報失敗，控制維持 HOLD | Scene adapter 階段 |
| `/uav/scene/start` — `geometry_msgs/msg/PoseStamped` | `uav_scene_bridge` | planner、recorder、visualization | `isaac_world`；stamp 必須與 obstacles/goal 的場景快照相同 | `R/TL/K1`；每次成功產生／更新時發布一次 | 必須與 obstacles 和 goal 屬於相同 episode 與快照 | 不一致的快照不得發布；generation 失敗並維持 HOLD | Scene adapter 階段 |
| `/uav/scene/goal` — `geometry_msgs/msg/PoseStamped` | `uav_scene_bridge` | planner、tracker、recorder、visualization | `isaac_world`；stamp 必須與場景快照相同 | `R/TL/K1`；每次成功產生／更新時發布一次 | 必須與 obstacles 和 start 屬於相同 episode 與快照 | 不一致的快照不得發布；沒有相符 goal 時 planner 不得執行 | Scene adapter 階段 |
| `/uav/scene/episode_id` — `std_msgs/msg/String` | `uav_scene_bridge` | 所有 episode 參與元件與 recorder | 不含 frame/header；相同 ID 也會出現在 `EpisodeState` 與 service/action 資料中。應以帶 stamp 的場景快照關聯，不使用 DDS 收件時間 | `R/TL/K1`；每個接受的 episode 發布一次，可選擇在值變更時重發 | 接受的 episode 必須有非空且唯一的 ID | ID 為空或重複時拒絕 generation；不啟用新場景 | Scene adapter 階段 |
| `/uav/scene/generate` — `uav_interfaces/srv/GenerateEpisode` | Server：`uav_scene_bridge` | Client：episode coordinator／`uav_bringup` composition | 不含 frame/header；response ID 用來關聯後續帶 stamp 的 `isaac_world` 快照 | 預設 service QoS；由 request 觸發，同時最多一個 active call | 只允許從 `IDLE` 或 reset 完成狀態呼叫；驗證 seed/count；`enable_recording` 只是請求，不代表錄製已開始 | 回傳 `success=false` 和詳細原因；不得發布部分場景，並維持或返回 HOLD | Scene adapter 階段 |

## Planning 規約

Phase 0 明確要求第一個里程碑的 planner/PX4 paths 使用 `px4_ned`。`coordinate_frames.md`
中的決策目前不允許使用保留的 `map` 替代方案。

Phase 3 實作 `path_raw`、`path_simplified`、
`path_bspline_candidate`, `bspline_valid`, `path`, and `status` with `R/TL/K1`
QoS。它要求帶 stamp 的 `isaac_world` inputs 相互匹配，且只有正規化內容改變時才重新規劃。
失敗時發布空的 `px4_ned` paths 和 `bspline_valid=false` 以清除舊資料，絕不發布非空且不安全的
final path。B-spline 只負責產生 candidate；所有 control endpoints 都維持停用。

| 精確名稱與型別 | 擁有者 | 訂閱端 | Frame 與 timestamp | QoS / 頻率 | 狀態限制 | 失敗行為 | 未來階段 |
|---|---|---|---|---|---|---|---|
| `/uav/planner/path_raw` — `nav_msgs/msg/Path` | `uav_navigation` A* adapter | simplifier、validator、debug visualization、recorder | Path 與每個 pose 都使用 `px4_ned`；stamp 使用規劃結果的 ROS 時間，pose stamps 必須相同或依同一時鐘明確排序 | `R/TL/K1`；每次規劃嘗試由事件觸發 | 只有 scene/frame 輸入一致後才發布診斷用 raw path；不可直接供 PX4 執行 | status 回報 empty/no-path；不發布 final path，並維持 HOLD | A* 擷取階段 |
| `/uav/planner/path_simplified` — `nav_msgs/msg/Path` | `uav_navigation` simplifier | B-spline candidate generator、final selector、validator、recorder | `px4_ned`；stamp 沿用目前規劃 transaction | `R/TL/K1`；每次成功簡化時由事件觸發 | 必須保留端點，並在 final selection 前通過連續線段驗證 | 失敗時 validator 可改用已驗證的 raw path；否則不產生 final path 並維持 HOLD | Planner 分解階段 |
| `/uav/planner/path_bspline_candidate` — `nav_msgs/msg/Path` | `uav_navigation` B-spline module | continuous validator、debug visualization、recorder | `px4_ned`；使用 candidate 產生時的 ROS stamp | `R/TL/K1`；只有要求平滑時才由事件觸發 | 僅供診斷的 candidate；不能只因發布了就視為可執行 | 非有限值、超出範圍、曲率、端點或連續淨空檢查失敗時，設定 `bspline_valid=false`；final path 改用已驗證的 A* | B-spline 階段 |
| `/uav/planner/path` — `nav_msgs/msg/Path` | `uav_navigation` final path selector/validator | A* follower candidate generator、recorder、visualization | `px4_ned`；stamp 使用 final validation/selection 時間 | `R/TL/K1`；事件觸發，並可選擇以 <=1 Hz latched 重發 | episode 狀態為 planning-ready 時，只發布完整通過驗證的 raw/simplified/B-spline 選擇結果 | 不得發布不安全或不完整的替代路徑；status 回報失敗且 command selection 維持 HOLD | Planner 整合階段 |
| `/uav/planner/bspline_valid` — `std_msgs/msg/Bool` | B-spline continuous validator | final selector、recorder、visualization | 不含 frame/header；適用於目前規劃 transaction/status 的 candidate | `R/TL/K1`；每個 candidate/validation result 發布一次 | 只有通過所有 continuous safety gates 才能設為 `true` | 值缺漏、有歧義或過期時視為 false；selector 改用已驗證的 A* fallback | B-spline 階段 |
| `/uav/planner/status` — `std_msgs/msg/String` | planning coordinator | episode coordinator、recorder、操作人員 UI | 不含 frame/header；文字包含 transaction/episode ID；權威時間仍以帶 stamp 的 paths/state 為準 | `R/TL/K1`；狀態轉換／錯誤時發布，規劃期間可選擇每秒發布一次 | 使用受控詞彙並附上說明；不可作為 command channel | 未知或錯誤狀態會阻擋 final path 接受，並強制或維持 HOLD | Planner 分解階段 |

## Candidate 與 selected control 規約

每個 `TwistStamped` command 的 `header.frame_id` 都必須是 `px4_ned`；linear velocity 使用 NED
座標，angular Z 遵循文件定義的 NED yaw-rate 慣例。header stamp 不可省略。Phase 6 mux input
QoS 為 reliable/volatile/keep-last 5，預設輸出頻率為 50 Hz。資料新鮮度依 mux 收到訊息的時間
判定，不採用 candidate 可自行指定的 header stamp；每個來源的 timeout 都可設定。

| 精確名稱與型別 | 擁有者 | 訂閱端 | Frame 與 timestamp | QoS / 頻率 | 狀態限制 | 失敗行為 | 未來階段 |
|---|---|---|---|---|---|---|---|
| `/uav/control/astar_command` — `geometry_msgs/msg/TwistStamped` | A* path follower only | command multiplexer, recorder | `px4_ned`; ROS clock at command computation | `R/V/K5`; 20 Hz while selected-capable | Only after a valid final path and active episode; bounded finite values | Stale/nonfinite/out-of-bounds command is rejected; mux selects HOLD | Follower phase |
| `/uav/control/joystick_command` — `geometry_msgs/msg/TwistStamped` | joystick node only | command multiplexer, recorder | `px4_ned`; ROS clock at input processing | `R/V/K5`; 20 Hz while teleop enabled | Requires fresh joystick input and deadman; cannot arm/PX4-publish directly | Timeout/deadman release publishes or selects HOLD, then candidate becomes stale | Teleop migration phase |
| `/uav/control/navrl_command` — `geometry_msgs/msg/TwistStamped` | bounded NavRL inference node only | command multiplexer, recorder | `px4_ned`; ROS clock at inference completion | `R/V/K5`; target 20 Hz, bounded by validated inference latency | Deployment/inference only; model loaded/validated, fresh observation, active episode | Late/invalid inference is discarded; never reuse stale output; mux selects HOLD | NavRL deployment phase |
| `/uav/control/hold_command` — `geometry_msgs/msg/TwistStamped` | safety controller only | command multiplexer, recorder | `px4_ned`; ROS clock each safety cycle | `R/V/K5`; 20 Hz whenever graph is control-capable | Always available before any non-HOLD source; bounded zero/position-hold semantics finalized with PX4 adapter | Loss of safety heartbeat blocks PX4 output rather than falling through to another candidate | Safety-controller phase |
| `/uav/control/selected_command` — `geometry_msgs/msg/TwistStamped` | `control_mux` only | Phase 6 offline plant; future PX4 output node and recorder | `px4_ned`; stamp is always the mux ROS clock at publication, never copied from a candidate | `R/V/K5`; configurable, default 50 Hz | Exactly one selected source; each candidate and final selected command are independently validated | Any stale/invalid selected source or validator failure produces internal exact-zero HOLD; no automatic movement-source failover | Phase 6 offline mux |
| `/uav/control/source` — `std_msgs/msg/String` | `control_mux` only | future PX4 output node, episode state owner, recorder, UI | No frame/header; published with every mux cycle | `R/V/K5`; default 50 Hz | Exact values `HOLD`, `ASTAR_EXPERT`, `HUMAN_JOYSTICK`, `NAVRL_POLICY`; startup is `HOLD` | Unknown request is rejected and fail-closes to latched HOLD | Phase 6 offline mux |
| `/uav/control/mux_status` — `uav_interfaces/msg/ControlMuxStatus` | `control_mux` only | test monitor, recorder, operator UI | `header.frame_id=px4_ned`; mux ROS clock | `R/V/K5`; default 50 Hz | Reports request/active source, HOLD reason, handoff time, source health/age, bounds and transition count | Diagnostics never authorize motion; contradictions fail closed | Phase 6 offline mux |
| `/uav/control/set_source` — `uav_interfaces/srv/SetControlSource` | Server: `control_mux` | Client: offline harness; future episode coordinator/UI | No request stamp; service handling uses mux receipt time and current registry health | Default service QoS | Only canonical sources accepted; movement-to-movement changes honor dwell and HOLD barrier; a fresh explicit request is required after a latched fault | Rejected requests return `accepted=false`; invalid requests select `HOLD_INVALID_SOURCE` | Phase 6 offline mux |

各元件只能發布自己負責的 candidate 或 selected topic：A* follower、joystick node、NavRL policy、
safety controller 和 mux 各自遵守此 ownership。未來 `uav_px4_control` 中的 PX4 output node
是唯一可發布實際 `/fmu/in/*` command topics 的 node。Planner、joystick、recorder、policy、scene、
camera 和 mux nodes 都不得發布這些 topics。Phase 6 只實作 candidate arbitration 和
`selected_command`，不包含 PX4 output publisher。

## Episode 規約

| 精確名稱與型別 | 擁有者 | 訂閱端 / clients | Frame 與 timestamp | QoS / 頻率 | 狀態限制 | 失敗行為 | 未來階段 |
|---|---|---|---|---|---|---|---|
| `/uav/episode/state` — `uav_interfaces/msg/EpisodeState` | episode coordinator | all packages, recorder, UI | Nonspatial header uses empty `frame_id` by contract; ROS stamp is state-transition/heartbeat time | `R/TL/K1`; on transition and 2 Hz heartbeat while active | One monotonic lifecycle per `episode_id`; phase and control source use controlled values; terminal success/collision immutable | Illegal transition produces failure/abort detail and HOLD; no success claim from incomplete cleanup | Episode orchestration phase |
| `/uav/episode/run` — `uav_interfaces/action/RunEpisode` | Action server: episode coordinator | Clients: operator/test harness | No frame/header in action fields; action acceptance time and feedback-associated `EpisodeState` provide ROS-time context | Default action QoS; one active goal, feedback target 2 Hz and on transitions | Validate episode ID/controller source/options before acceptance; cancellation always supported; `enable_bspline` requests validation, not bypass | Reject invalid goal; cancel/abort selects HOLD, stops recording safely, cleans up, and returns `success=false` with detail | Episode orchestration phase |

## Camera 規約

| Exact name and type | Owner | Subscribers | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/fpv/image_raw` — `sensor_msgs/msg/Image` | `uav_camera_bridge` FPV publisher | NavRL inference, recorder, visualization | `uav_fpv_camera`; capture simulation-time stamp | `BE/V/K5`; nominal 10 Hz, configurable/declared | Publish only complete supported encodings with dimensions/step consistent; same stamp as FPV CameraInfo | Drop corrupt/incomplete frame, increment diagnostics; never fabricate or block control thread | Camera bridge phase |
| `/uav/fpv/camera_info` — `sensor_msgs/msg/CameraInfo` | `uav_camera_bridge` FPV calibration owner | same FPV consumers | `uav_fpv_camera`; exact matching image stamp | `BE/V/K5`; with every image at nominal 10 Hz | Calibration dimensions/model must match image and active camera configuration | Drop unmatched pair; policy/recorder rejects image without matching calibration | Camera bridge phase |
| `/uav/observer/image_raw` — `sensor_msgs/msg/Image` | `uav_camera_bridge` observer publisher | recorder, visualization, optional policy | `uav_observer_camera`; capture simulation-time stamp | `BE/V/K5`; nominal 10 Hz, configurable/declared | Same completeness/encoding rules; pairing with FPV uses episode ID + ROS stamp/tolerance | Drop corrupt frame and report diagnostic; no half-pair dataset row | Camera bridge phase |
| `/uav/observer/camera_info` — `sensor_msgs/msg/CameraInfo` | `uav_camera_bridge` observer calibration owner | same observer consumers | `uav_observer_camera`; exact matching image stamp | `BE/V/K5`; with every image at nominal 10 Hz | Calibration/configuration must match active observer attachment mode | Drop unmatched pair and report diagnostic | Camera bridge phase |

## Vehicle state 規約

| Exact name and type | Owner | Subscribers | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/vehicle/pose` — `geometry_msgs/msg/PoseStamped` | PX4 telemetry adapter in `uav_px4_control` | planner/tracker, safety, recorder, UI | `px4_ned`; source sample converted into common ROS clock | `BE/V/K5`; expected 20–50 Hz, configured to telemetry rate | Publish only finite, valid local pose after origin/frame readiness | Stale/invalid pose marks vehicle state unhealthy, rejects non-HOLD control, and blocks output activation | PX4 telemetry phase |
| `/uav/vehicle/twist` — `geometry_msgs/msg/TwistStamped` | PX4 telemetry adapter | tracker, safety, recorder | `px4_ned`; same source-time conversion policy | `BE/V/K5`; expected 20–50 Hz | Finite velocity with declared NED/angular convention; episode/frame ready | Stale/invalid twist blocks non-HOLD activation and is surfaced in episode status | PX4 telemetry phase |
| `/uav/vehicle/odometry` — `nav_msgs/msg/Odometry` | PX4 telemetry adapter | planner/tracker, safety, recorder, UI | `header.frame_id=px4_ned`, `child_frame_id=base_link`; common ROS stamp | `BE/V/K5`; expected 20–50 Hz | Pose/twist/covariance internally consistent; transform contract validated | Stale/inconsistent odometry marks unhealthy; no extrapolated sample may silently command flight | PX4 telemetry phase |

## Phase 4 trajectory 規約

Phase 4 由事件觸發，使用 reliable transient-local depth-one QoS。此階段嚴格位於 Phase 3
最終選擇之後，不屬於 command path。

| 精確名稱與型別 | 擁有者 | 輸入 / 來源 | Frame 與 timestamp | 接受條件與失敗行為 |
|---|---|---|---|---|
| `/uav/trajectory/candidate` — `uav_interfaces/msg/TimedTrajectory` | `uav_navigation` trajectory parameterizer | 只讀取 `/uav/planner/path`；忽略 pose stamps 與 orientations | Header 和 `source_path_frame` 都是 `px4_ned`；output header 使用目前 ROS 時間 | 只發布有限且結構正確的 candidate。通過獨立驗證後才可設 `valid=true`；遭拒的有限 candidate 帶有 `valid=false` 和 status。 |
| `/uav/trajectory/valid` — `std_msgs/msg/Bool` | 獨立的 trajectory validation result publisher | 目前唯一的 path 嘗試 | 不含 header；與目前 candidate/status transaction 配對 | 每次嘗試不同的 path 都發布結果。缺漏、過期或遭拒的結果一律不得解讀為 true。 |
| `/uav/trajectory/status` — `std_msgs/msg/String` | Trajectory parameterizer | 目前唯一的 path 嘗試 | 不含 header；使用受控的 pipe-delimited 欄位 | 回報成功／拒絕、計數、持續時間、時間縮放、動態量最大值和明確拒絕原因。 |

`TimedTrajectory` 的 positions 必須與移除相鄰重複點後的來源 path 完全相同。Time 和 arc length
必須是有限值且嚴格遞增。此 node 不讀取 raw/simplified/candidate planner topics、vehicle state、
joystick、NavRL、simulator 或 PX4 topics。Phase 4 的任何 owner 都不得發布 `/fmu/in/*`。

## Phase 5 offline tracking 規約

Phase 5 只讀取已接受的 Phase 4 candidate/validity pair 和 offline `px4_ned` odometry。其
reliable volatile 20 Hz 輸出是 ROS 層級的 controller candidates 與 diagnostics，不是 PX4
setpoints。

| 精確名稱與型別 | 擁有者 | 說明與 frame | 失敗行為 |
|---|---|---|---|
| `/uav/control/astar_command` — `geometry_msgs/msg/TwistStamped` | `trajectory_follower_node` | `px4_ned`；linear X/Y/Z 為北／東／下速度，angular Z 為 NED yaw rate，angular X/Y 為零 | 輸入缺漏、過期、false、frame 錯誤、非有限值、時間跳變、誤差過大、terminal timeout 或 validator 失敗時，會選擇精確零值 HOLD 並提供原因 |
| `/uav/control/astar_reference_pose` — `geometry_msgs/msg/PoseStamped` | `trajectory_follower_node` | `px4_ned` 中目前插值後的參考位置／yaw | 無有效參考值可取樣時不發布 |
| `/uav/control/astar_reference_twist` — `geometry_msgs/msg/TwistStamped` | `trajectory_follower_node` | 目前插值後的 NED 速度與 yaw rate | 無有效參考值可取樣時不發布 |
| `/uav/control/astar_tracking_status` — `uav_interfaces/msg/TrajectoryTrackingStatus` | `trajectory_follower_node` | 帶 stamp 的 `px4_ned` state、gates、誤差、飽和旗標、diagnostics 與原因 | 明確回報 waiting/HOLD/terminal 狀態；不會默默重用過期證據 |

Candidate command、selected command 和 PX4 output command 分屬三個不同的 ownership layers。Phase 5
只實作第一層：在 `/uav/control/astar_command` 發布經驗證的 A* follower candidate。內部的純量
`selected_command` field 只表示同一 follower cycle 中，從未限幅計算結果選出的有界 candidate；
它不代表 mux arbitration。Phase 6 已實作負責 `/uav/control/selected_command` 的 source mux；
只有未來獨立的 `uav_px4_control` adapter 可以將該輸出映射到 `/fmu/in/*`。Phase 6 graph 尚未
包含該 PX4 output layer。

收件時間加上 `trajectory_start_delay_s` 定義本機 tracking epoch；trajectory timestamps 維持
相對時間。內容完全相同的重複 trajectory 不會重設 epoch。Control time 倒退或相等時會 fail
closed；發生倒退後，必須重新同步新鮮的 trajectory、validity 和 odometry。

## Phase 6 offline control mux 規約

Mux 會訂閱四個 candidate topics，但移動控制權仍採互斥方式管理。啟動時和收到明確的 `HOLD`
請求時，都使用內部精確零值 command。切換到移動來源前，candidate 必須是新鮮、有限且有界的
`px4_ned` 資料。移動來源彼此切換時會先經過精確零值 HOLD barrier；若目標來源轉為不健康，
handoff 就會取消。選取來源發生 fault 後，HOLD 會鎖存，訊息恢復本身不會解除鎖存狀態。

獨立的 selected-command validator 會檢查水平、垂直與總速度、加速度、yaw-rate、yaw-acceleration、
frame、有限值及時間單調性。Safety HOLD 會立即輸出精確零值，不受一般 rate limits 延遲。未選取
來源的 fault 不會取代仍健康的 active source；mux 也不會自動切換到另一個移動來源。

## 全域失敗與生命週期規則

- 每個 candidate 和 selected command 都必須帶有 timestamp。過期 candidate 會被拒絕；來源選擇無效或 selected command 遺失時，系統進入 HOLD。
- 持續保存的 scene/path/state 資料不可當成新鮮的控制輸入。
- 元件完成必要的 cleanup 和 recording finalization 前，不得發布成功的 terminal state。
- QoS 不相容、frame 不符、缺少 `/clock`、時間跳變或 episode ID 不符，都必須明確呈現為健康狀態失敗；不得因此無限期沿用最後一筆資料。
- 所有頻率都是目標規約；實作時必須將其設為參數，並提供診斷量測。
