# ROS 2 開發者指令

## 目的與安全邊界

Repository 根目錄的 `./uav` wrapper 將重複執行的環境檢查、build、test、inspection 和
offline checks 整合成可重現的單一入口。它是非飛行的開發工具，不會啟動 Isaac Sim、Pegasus、
PX4、Micro XRCE-DDS、cameras、recorders、OFFBOARD mode、arming 或 takeoff。Phase 6 和 7
只會依需求啟動 synthetic inputs、offline mux、mapping/gate diagnostics、既有 follower 和
fixed-step plant；不會發布 `/fmu/in/*` commands。

執行 `./uav help` 可查看精簡的指令列表。

## 隔離環境

ROS commands 透過 `env -i` 執行，只保留 `HOME`、`USER`、`PATH=/usr/bin:/bin`、
`LANG=C.UTF-8`、選用的 `TERM` 和 offline timeout。Wrapper 會 source
`/opt/ros/jazzy/setup.bash`、確認 `/usr/bin/python3`，並在需要時只 source 本 repository 的
`ros2_ws/install/setup.bash`。它不會 source 舊版 `uav_ros2_ws`，也不會繼承 virtualenv、
pyenv、AMENT 或 COLCON overlays。

每個 clean command 都會印出選用的 Python、Python 版本、ROS distribution、workspace path
和 overlay state。

## 指令

| 指令 | 行為 |
|---|---|
| `./uav status` | 唯讀檢視 Git 與 workspace 目錄狀態 |
| `./uav doctor` | Dependency, contamination, branch, overlay, and topic checks |
| `./uav build` | Clean Jazzy `colcon build --symlink-install` |
| `./uav test` | Clean sourced `colcon test` and verbose results |
| `./uav verify` | Doctor, build, tests, packages, interfaces, imports, and scans |
| `./uav offline` | 互動式啟動 offline planner，預設不設 timeout |
| `./uav offline-check` | 有限時、保留 log 且結果確定的 offline 驗證 |
| `./uav topics` | ROS graph listing and `/fmu/in/*` rejection |
| `./uav interfaces` | Custom definitions and canonical ROS topic/service names |
| `./uav mux-check` | Nominal four-source arbitration and HOLD handoffs |
| `./uav mux-safety-check` | 驗證來源過期鎖存、明確復原及內部 HOLD |
| `./uav control-stack-check` | Scene through follower, mux and offline plant |
| `./uav px4-map-check` | Pure and live diagnostic NED candidate mapping |
| `./uav px4-gate-check` | Synthetic failsafe, latch, reset, and re-enable |
| `./uav px4-boundary-check` | Scene through mux to `safe_to_forward` |
| `./uav shell` | 開啟新的乾淨互動式 shell，提示字元為 `[uav-ros2]` |

日常使用範例：

```bash
./uav doctor
./uav build
./uav test
./uav verify
./uav offline-check
./uav trajectory-check
./uav pipeline-check
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
./uav px4-map-check
./uav px4-gate-check
./uav px4-boundary-check
```

## Offline modes 與 launch arguments

`./uav offline` 會將 terminal 直接交給 `ros2 launch`；它沒有 timeout，可按 Ctrl+C 停止。
`./uav offline-check` 使用 Phase 3 harness，檢查結構化 success marker、source、validity、
path 數量與 frame，並將含 timestamp 的 log 寫入 Git 忽略的 `run_logs/`。預設安全 timeout
為 20 秒：

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=30 ./uav offline-check
```

任一 subcommand 後方的 arguments 都會傳給 launch file：

```bash
./uav offline use_sim_time:=false
./uav offline-check enable_bspline:=true fixture:=bspline-safe-single-obstacle
./uav offline-check enable_bspline:=true fixture:=bspline-rejected-corner-cut
./uav offline-check enable_bspline:=false fixture:=bspline-disabled
```

Phase 4 增加兩個有限且不飛行的檢查。`trajectory-check` 將已驗證的 `px4_ned` path 直接
發布給 parameterizer；`pipeline-check` 發布固定場景，並驗證完整的 A* / B-spline-or-fallback /
timed-trajectory chain。兩者都在 clean Jazzy environment 執行、轉交 launch arguments、掃描
`/fmu/in/*`，並寫入含 timestamp 的忽略 logs：

```bash
./uav trajectory-check fixture:=straight-line
./uav trajectory-check fixture:=impossible-config-rejection
./uav trajectory-check fixture:=wrong-frame
./uav pipeline-check enable_bspline:=true
./uav pipeline-check enable_bspline:=false
```

Standalone fixtures 名稱包括 `straight-line`、`phase3-bspline`、`sharp-bend`、
`high-curvature`、`duplicate-adjacent`、`two-point`、`invalid-one-point`、`nonfinite`、
`yaw-wrap`、`jerk-scaling`、`impossible-config-rejection` 和 `wrong-frame`。

可用的邊界 fixtures 包括 `short-two-point-path`、`three-point-path`、
`duplicate-control-point-path`、`self-intersection-candidate` 和
`curvature-limit-rejection`。Build 後可使用以下命令重現六場景 geometric report：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run uav_navigation geometric_path_comparison
```

## Phase 5 offline tracking checks

Phase 5 仍不執行飛行。Follower 只會在 `px4_ned` 發布 ROS-level
`/uav/control/astar_command` candidate 與 reference/status topics；launch files 不包含 PX4、
Isaac Sim、Pegasus、XRCE、arming、OFFBOARD 或 `/fmu/in/*` 路徑。

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
```

`tracking-check` 將固定 timed trajectory 傳給 follower 和 deterministic first-order plant，
直到進入 `GOAL_HOLD`。`tracking-safety-check` 預設使用 `stale-odometry`，並要求精確的零值
HOLD 與預期原因。`full-pipeline-check` 驗證固定場景 -> A* -> 接受 B-spline 或 A* fallback
-> timed trajectory -> follower -> plant 的完整 chain。三個 commands 都有執行期限、會掃描
`/fmu/in/*`，並將忽略 logs 寫入 `run_logs/`。

常用 safety fixtures：

```bash
./uav tracking-safety-check fixture:=stale-odometry
./uav tracking-safety-check fixture:=invalid-validity-flag
./uav tracking-safety-check fixture:=wrong-odometry-frame
./uav tracking-safety-check fixture:=excessive-tracking-error
```

完整的確定性 fixtures 名稱包括 `straight-trajectory`、`phase3-bspline-accepted`、
`astar-fallback`、`sharp-dynamically-valid`、`start-position-offset`、
`constant-horizontal-disturbance`、`duplicate-trajectory-message`、`stale-odometry`、
`stale-trajectory-validity`、`invalid-validity-flag`、`wrong-odometry-frame`、
`nonfinite-odometry`、`backward-time-jump`、`command-speed-saturation`、
`command-acceleration-saturation`、`excessive-tracking-error`、
`successful-goal-settling`、`terminal-not-reached`、`yaw-wrap-crossing` 和
`invalid-command-rejection`。

Build 後可使用以下命令重現八項 pure fixture 表格：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run uav_navigation tracking_comparison
```

## Phase 6 offline mux checks

Phase 6 只負責 ROS-level arbitration。`mux-check` 依序要求
`ASTAR_EXPERT -> HUMAN_JOYSTICK -> NAVRL_POLICY -> HOLD`, requires an
exact-zero HOLD barrier between movement sources, and verifies one selected
publisher. Its monitor waits for at least three recent, strictly monotonic
candidate heartbeats plus mux-reported health before each movement-source
request, and retains ACTIVE observations across asynchronous service-response
ordering. `mux-safety-check` forces selected-source staleness, checks the
latched HOLD cannot auto-recover, then performs an explicit recovery request;
it also verifies an invalid external HOLD cannot disable internal HOLD.
`control-stack-check` 將實際 Phase 5 follower 接到 mux，並讓 offline plant 只使用
`/uav/control/selected_command`。

```bash
./uav mux-check
./uav mux-safety-check
./uav control-stack-check enable_bspline:=true
```

所有 commands 都有執行期限、會在 `run_logs/` 寫入忽略 logs、掃描 live graph 是否出現
`/fmu/in/*`，而且不會啟動 PX4、Isaac Sim、joystick hardware 或 NavRL runtime/model。

## Phase 7 offline PX4 boundary checks

Phase 7 將 mux 選出的 `px4_ned` velocity/yaw-rate command 轉成自訂 diagnostic candidate。
`px4-map-check` 會在 ROS graph 前執行 pure timestamp/mapping fixtures。`px4-gate-check` 證明
輸出預設停用、必須明確 enable、遇到 synthetic failsafe 會 fail closed、telemetry 恢復後仍會
鎖存，並要求 disable/reset/re-enable。`px4-boundary-check` 執行完整 scene-to-gate graph，並在
先觀察到 `SAFE_TO_FORWARD` 後刻意注入相同故障。

```bash
./uav px4-map-check
./uav px4-gate-check
./uav px4-boundary-check
```

這些 commands 只發布 `/uav/px4/*` diagnostics 和 `/uav/test/px4/*` 下的 synthetic state。
Boolean permission 不代表 PX4 publisher，也不代表飛行授權。

## Phase 8 僅限 SITL 的 stream checks

預設 offline check 不依賴 PX4，且必須在啟動 PX4 前通過：

```bash
./uav px4-stream-offline-check
```

它會執行 35 項 pure/unit/static tests 與 20-fixture stream matrix、強制執行兩個 topic 的
publisher allowlist，並拒絕任何包含 `/fmu/in/*` 的 live graph，因為執行此 gate 時 PX4 不得
運行。

標準七個 package 的 build 刻意不 source legacy workspace。第一次 live check 前，請將乾淨的
外部 `px4_msgs` source build 到此 workspace，不要複製或修改該 source：

```bash
./uav build --base-paths src \
  /home/noel_614420090/uav_ros2_ws/src/px4_msgs
```

請在不同的 `tmux` windows 手動啟動已稽核的 PX4 SIH target 和 Micro XRCE-DDS。Wrapper 不會
代為啟動：

```bash
tmux new-window -t astar -n xrce
tmux send-keys -t astar:xrce \
  'MicroXRCEAgent udp4 -p 8888' C-m

tmux new-window -t astar -n px4
tmux send-keys -t astar:px4 \
  'ninja -C /home/noel_614420090/PX4-Autopilot/build/px4_sitl_default sihsim_quadx' C-m
```

`sihsim_quadx` 是本機檢查過的 PX4 內建 SITL simulation target，不會啟動 Isaac Sim 或
Gazebo。等待 PX4 啟動並建立 DDS endpoints 後，執行唯讀前置檢查：

```bash
./uav px4-sitl-doctor
```

Doctor 不會建立 PX4 input publisher。它要求本機有預期的 SITL process、
`MicroXRCEAgent udp4 -p 8888`、四個 telemetry publishers、兩個 PX4 input subscribers、相容
的 endpoint QoS、disarmed state、OFFBOARD 未啟用、沒有 critical failsafe，且沒有
`VehicleCommand` publisher。它要求連續三次取得完整 readiness snapshots，避免將 Agent
重啟後 DDS discovery 收斂期間誤認為穩定就緒。

Doctor 成功後才執行：

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=30 ./uav px4-sitl-stream-check
```

有限 launch 會以停用狀態開始，透過既有 mux 和 Phase 7 mapper/gate 傳送零值 candidate，明確
啟用 streamer、確認至少 40 組 message pairs 與 timing，接著停用並證明 publication 已停止。
選用的 mapping-only fixtures 為 `fixture:=north-0.10`、`east-0.10`、`down-0.10` 和
`yaw-rate-0.10`；只有零值 fixture 成功後才能分開執行它們。

Live launch 只將 `telemetry_timeout_s` 覆寫為 0.75 s，因為稽核到的 PX4 status/flags topics
測得頻率約 1.7--1.9 Hz，最大間隔達 0.575 s。Candidate timeout 維持 0.25 s、gate timeout
維持 0.50 s、Phase 7 limits 不變，publish-gap 安全門檻仍為 0.20 s。Monitor 會回報 pair
數量、平均頻率、最小/最大間隔、RMS jitter 以及第一筆/最後一筆 timestamp。

Phase 8 不會發布 `/fmu/in/vehicle_command`、要求 OFFBOARD、arm、disarm、起飛或降落，也
不會啟動 Isaac Sim 或控制實體載具。

## 受保護的 PX4 SITL 飛行里程碑

本機 XRCE Agent 和 PX4 SIH 已在執行時，執行：

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=150 ./uav px4-sitl-flight-check
```

不同於 Phase 8 stream fixture，此 command 會刻意進入 OFFBOARD、arm、執行設定的 1.5 m
A*/B-spline 任務，並要求 PX4 AUTO_LAND。若初始狀態不是 SITL、已 armed、OFFBOARD 已啟用、
進入 failsafe 或存在其他 command owner，wrapper 會拒絕執行。完整證據與限制見
`px4_sitl_flight_milestone.md`。

## 為什麼 shell 會建立新的 shell

已執行的 script 無法安全修改 parent shell 的 environment。因此 `./uav shell` 使用 `exec`
開啟 child Bash process，並 source Jazzy 及（若存在）此 workspace overlay。按 Ctrl+D 可正常
離開並回到原 terminal，不會污染原本的 shell。

啟動摘要會列出 `ROS_DISTRO`、Python、workspace、overlay state 和 `px4_ned` planning frame。

## 疑難排解

若 pyenv 或 virtual environment 選到不相容的 Python，不要將 ROS dependencies 安裝到該環境。
執行 `./uav doctor`；clean commands 會刻意清除 pyenv shims、`PYENV_VERSION`、`VIRTUAL_ENV`
與繼承的 prefix paths。

若找不到 `install/setup.bash`，請執行：

```bash
./uav build
```

Wrapper 不會刪除 `build/`、`install/` 或 `log/`。只有在明確需要時，才手動移除或搬移產生的
artifacts。

## 選用 alias

Wrapper 不會修改 shell startup files。若有需要，可手動加入：

```bash
alias uav="$HOME/uav-project/uav"
```

之後開啟新 shell 即可使用 `uav verify`、`uav offline` 和 `uav shell`。
# ML 和 BC 基準指令

這些指令使用呼叫端的 Python environment，不會 source ROS 2：

```bash
./uav ml-doctor
./uav bc-train --help
./uav bc-train
./uav bc-eval --help
./uav bc-eval --episodes 20
./uav ml-test
```

`bc-train` 不依賴 ROS，也不會啟動 Isaac 或 PX4。`bc-eval` 會啟動既有的 headless Isaac
固定高度城市 camera environment，並在與 expert seeds 不重疊的 seeds 上交由 BC policy
獨占控制。artifact 路徑與規約請參閱
[`bc_baseline.md`](bc_baseline.md) for artifact paths and contracts.
