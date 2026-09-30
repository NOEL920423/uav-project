# Phase 6 control mux 回歸規約

## 安全不變條件

每個發布的 selected command 都必須有 mux-clock stamp、`px4_ned` frame、有限欄位、零值
`angular.x/y`，並通過獨立的 selected-command validator。一般移動必須符合 total/horizontal/
vertical speed、acceleration、yaw-rate 和 yaw-acceleration limits。所有非移動輸出都必須是
精確零值，`active_source=HOLD`，且附有非空原因。

只有明確啟用的 movement source 可以被轉送。未選取的健康 candidate 不得影響輸出；未選取的
不健康 candidate 也不得中斷目前 active source。任何 fault 都不得自動切換 movement source。
Fault 鎖存後，必須取得新鮮資料並收到明確有效的 service request，才能恢復移動。

Movement source 互相切換時，必須至少經過設定時間的精確零值 switch barrier。Barrier 結束後
重新驗證目標 source。任何 graph 或 module 都不得將 selected command 發布或 remap 至
`/fmu/in/*`。

## 確定性 fixtures

| # | 測試案例 | 預期 service/state/output 順序 |
|---:|---|---|
| 1 | 啟動時沒有來源 | `HOLD_STARTUP`，精確零值 |
| 2 | 選取新鮮的 A* | 接受請求、`ACTIVE_ASTAR_EXPERT`，只輸出有界 A* command |
| 3 | active A* 過期 | `HOLD_STALE_SOURCE`，精確零值 |
| 4 | 過期 fault 鎖存 | 只有新鮮 candidate 仍維持 `HOLD_LATCHED_FAULT` |
| 5 | 明確要求 A* 復原 | 接受新鮮請求，恢復 active A* |
| 6 | A* 切換至 joystick | 接受請求，A* -> barrier -> joystick |
| 7 | 觀察切換 barrier | 整段設定期間都輸出精確零值 |
| 8 | handoff 期間目標來源過期 | barrier -> fail-closed stale HOLD |
| 9 | joystick 切換至 NavRL | 接受請求，joystick -> barrier -> NavRL |
| 10 | 未知來源 | 拒絕、`HOLD_INVALID_SOURCE`、精確零值 |
| 11 | 選取來源 frame 錯誤 | `HOLD_WRONG_FRAME`，鎖存精確零值 |
| 12 | 選取來源含非有限值 | `HOLD_INVALID_COMMAND`，鎖存精確零值 |
| 13 | 選取來源速度超限 | 拒絕／fail-closed，維持有界規約 |
| 14 | candidate stamp 非單調 | `HOLD_INVALID_COMMAND`，鎖存零值 |
| 15 | node 時間倒退 | 清除歷史資料，進入 `HOLD_TIME_JUMP` |
| 16 | 最短 dwell 時間 | 過早的請求遭拒，且不混用控制權 |
| 17 | 重複要求目前來源 | 冪等處理，不增加 transition 計數 |
| 18 | 明確要求 HOLD | 立即接受精確的 `HOLD_REQUESTED` |
| 19 | 外部 HOLD 無效 | 內部 HOLD 仍可使用且有效 |
| 20 | 多個來源同時輸入 | 只轉送選定來源 |
| 21 | 未選取來源過期 | 健康的 active source 持續運作 |
| 22 | 選取來源加速度限制 | 輸出變化量符合 `1.5 m/s^2` |
| 23 | 選取來源 yaw-acceleration 限制 | 輸出變化量符合 `2.0 rad/s^2` |
| 24 | follower 經 mux 和 plant 運作 | 僅使用 A* candidate、僅由 mux 擁有輸出，最後進入 `GOAL_HOLD` |

每個 fixture 都記錄 request/response、requested 和 active source、command sequence、HOLD
原因/週期、source age、transition count、鎖存與復原行為，以及預期和實際觀察到的 terminal
state。

## 必要的 pure coverage

- 精確的來源識別名稱、topic 對應、timeout 查找與未知來源拒絕；
- 設定值的有限性、正值、相容性與 boolean 型別；
- candidate 尚未收到、新鮮、位於邊界年齡、過期、frame 錯誤、非有限、超限及時間非單調等分類；
- 啟動 HOLD、明確 HOLD、直接啟用、冪等請求、dwell 拒絕、移動來源切換 barrier、目標重新驗證與取消 handoff；
- active fault 鎖存、新資料不能自動復原、明確復原、時間倒退重設與未選取來源隔離；
- 水平、垂直、總速度、加速度、yaw-rate 與 yaw-acceleration 限制，以及來源歷史資料重設；
- 獨立 validator 的 ownership、frame、有限性、angular x/y、所有限制、timestamp、HOLD 數值／原因與 active state 一致性；
- 獨立驗證 fallback HOLD 與結構化 diagnostics；以及
- 全部 24 個確定性測試案例和 comparison table 產生流程。

## 必要的 ROS graph coverage

直接的有限 graph 只包含 synthetic candidate publishers、mux 和獨立 monitor。它會觀察 startup
HOLD、成功選擇 A*、完整的 movement-to-movement 零值 barrier、精確 source ownership、三個
output topics、service response fields，並確認沒有 `/fmu/in/*` publisher。要求 movement source
前，monitor 必須觀察至少三筆近期 candidate arrivals，其 stamps 嚴格遞增，並獨立確認 mux
回報該 source 健康。已送出的 service requests 和觀察到的 ACTIVE events 是不可撤銷的同步
證據：非同步 service response 前已發生的 activation，不能被較晚的 status sample 抹除。
Nominal graphs 遇到任何 stale-source 或鎖存 HOLD 都會立即失敗。

Safety graph 至少涵蓋 active-source 過期、fault latch、新資料不會自動復原、明確復原、錯誤
frame、非有限輸入、超限輸入、publisher stamps 非單調、handoff 期間 target 過期、外部
HOLD 失敗，以及 fail-closed internal HOLD。

Control-stack graph 只加入既有 offline scene、planner、trajectory parameterizer、A* follower、
selected-command mux、deterministic plant 和有限 monitor。Plant 只訂閱 selected command。它
必須觀察 A* candidate publication、接受 `ASTAR_EXPERT` selection、selected-command ownership、
有界且連續的 commands、tracking progression 和 terminal `GOAL_HOLD`。

必要的 wrapper gates：

```bash
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
```

Phase 2–5 的全部 wrappers 和 whole-workspace test suite 仍是必要項目。Static scans 必須證明
follower 只擁有
`astar_command`, synthetic joystick and NavRL fixtures own only their candidate
topics, the safety fixture owns only `hold_command`, only the mux owns
`selected_command`, pure modules contain no forbidden runtime import, and
tracked output contains no build, log, cache, bag, model, or dataset artifact.

## 完成範圍

通過此回歸規約只代表確定性的離線 ROS-level source arbitration 已成立。它不驗證 PX4
setpoint mapping、OFFBOARD、arming、實體 joystick、NavRL policy/runtime/model、Isaac Sim、
vehicle dynamics、真實擾動抑制或飛行能力。
