# PX4 output gate 回歸規約

Phase 7 停在診斷用的 permission boundary。`safe_to_forward=true` 只表示未來的 publisher
可能可以使用目前的 candidate；它不會發布或授權 PX4 uORB/ROS input topic。

## 必要證據

- a fresh, independently validated velocity-only candidate;
- a fresh Phase 6 mux status which is valid, not in HOLD, and names the same
  canonical active source;
- fresh synthetic PX4 telemetry with monotonically nondecreasing timestamps;
- connected, acceptable standby/armed and navigation values;
- passing preflight evidence, valid local position/velocity/odometry, and NED
  pose/velocity frames;
- no failsafe and no offboard-control-signal-loss evidence;
- an explicit output-enable request followed by one complete healthy cycle.

在此診斷 gate 中，Disarmed (`ARMING_STATE_STANDBY`) 且 OFFBOARD 尚未啟用的 telemetry
仍可視為健康。這是刻意設計：Phase 7 不決定 arming 或 mode-switch 順序，只判定 candidate
是否可安全交給未來另行審查的 Phase 8 publisher/sequencer。

## Fail-closed 與復原規則

啟動狀態為 `OUTPUT_DISABLED`。缺少輸入時使用明確的 `WAITING_*` states。Command 或
telemetry 過期、mapping 無效、mux HOLD、failsafe、時間倒退或 vehicle state 意外改變，
都會立即令 `safe_to_forward=false`。若當時已啟用輸出請求，fault 會鎖存；新資料不能
自行解除鎖存。復原步驟如下：

1. request disable/reset;
2. repair and revalidate all evidence;
3. request enable;
4. complete one additional healthy gate cycle.

此規約不包含自動啟用、重試、mode switch、arm request、setpoint streaming 或
`/fmu/in/*` publisher。
