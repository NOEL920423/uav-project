# Trajectory 回歸規約

本規約固定 Phase 4 離線 trajectory 行為。

## 輸入與輸出

- 輸入 topic：`/uav/planner/path`，type `nav_msgs/msg/Path`，frame `px4_ned`。
- Candidate topic：`/uav/trajectory/candidate`，type
  `uav_interfaces/msg/TimedTrajectory`.
- Validity topic：`/uav/trajectory/valid`，type `std_msgs/msg/Bool`。
- Status topic：`/uav/trajectory/status`，type `std_msgs/msg/String`。
- Input pose timestamps 和 orientations 沒有語意用途。
- 相鄰重複位置會移除；其他位置與高度保持不變，順序也不改變。

## 接受條件

每個被接受的 candidate 都必須符合以下條件：

- 至少有兩個 points，且所有數值欄位都是有限值。
- Time 從零開始並嚴格遞增。
- Arc length 從零開始並嚴格遞增。
- Geometry 必須與清理後的 source path 完全相同。
- Speed、longitudinal acceleration/deceleration、lateral acceleration、jerk、yaw rate 和
  yaw acceleration 都不得超過設定上限（允許文件所述的 numerical tolerance）。
- 必須遵守指定的起點與終點零速/非零速度。
- Yaw 為連續、未包裹的 NED heading `atan2(east, north)`。
- 只有獨立驗證通過後，才可發布 `valid=true`。

只要可取得相關資訊，拒絕結果都會指出失敗條件、point index、測量值與限制。Frame
錯誤、點數不足與非有限輸入都會以確定性方式拒絕。收到完全相同的 path 時，不會再次
計算或發布。

## 必要的確定性 fixtures

Standalone harness 支援以下 fixtures：`straight-line`、`phase3-bspline`、`sharp-bend`、
`high-curvature`、`duplicate-adjacent`、`two-point`、`invalid-one-point`、`nonfinite`、
`yaw-wrap`、`jerk-scaling`、`impossible-config-rejection` 和 `wrong-frame`。

`./uav trajectory-check` 執行 standalone graph。`./uav pipeline-check` 會讓固定的
Phase 3 場景經過 A*、可選的 B-spline selection 與 Phase 4 parameterization。兩者都在
乾淨的 ROS 2 Jazzy environment 執行、寫入 Git 忽略的 runtime log、掃描是否出現
`/fmu/in/*`、轉交 launch arguments，並在未觀察到預期結果時回傳非零狀態。

## 不包含的功能

此階段不包含 follower、controller、setpoint conversion、OFFBOARD 或 arming logic、
takeoff/landing 行為、simulator 整合、camera/recorder、NavRL、XRCE agent 或 PX4 input
publication。有效 candidate 只是分析資料，不代表獲准飛行。
