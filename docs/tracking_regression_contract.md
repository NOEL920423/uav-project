# Phase 5 離線 tracking 回歸規約

## 不變條件

每個 accepted trajectory 都必須使用 `px4_ned` frame、至少有兩個有限 points、relative
timestamps 嚴格遞增，且有相符的新鮮 true validity sample。接收時間加上設定延遲定義 epoch。
完全相同的重複 trajectories 不會重設 epoch。

每個一般 command 都必須是有限值、使用 `px4_ned`、`angular.x/y` 為零、符合 total、horizontal、
vertical、acceleration、yaw-rate 和 yaw-acceleration limits，並通過獨立 validator。所有非
一般輸出都必須是精確零值 HOLD，且附有非空原因。Phase 5 graph 不得包含 topic 以 `/fmu/in/`
開頭的 publisher。

Terminal success 要求抵達 final reference time，且 position、measured speed 和 wrapped yaw
tolerances 在整段 settle interval 中持續符合。未能穩定即 timeout 時，必須確定性地拒絕並進入
HOLD，不得回報成功。

## 確定性 fixtures

| # | 測試案例 | 預期結果 |
|---:|---|---|
| 1 | 直線 trajectory | tracking 成功並進入 `GOAL_HOLD` |
| 2 | 接受的 Phase 3 B-spline trajectory | tracking 成功並進入 `GOAL_HOLD` |
| 3 | A* fallback trajectory | tracking 成功並進入 `GOAL_HOLD` |
| 4 | 尖銳但動態上有效的 trajectory | 有界 tracking 成功 |
| 5 | 起始位置偏移 | feedback 收斂並成功 |
| 6 | 固定水平擾動 | tracking 有界成功，並量測誤差 |
| 7 | 重複 trajectory 訊息 | 只接受一次；epoch 不變 |
| 8 | 過期 odometry | 在設定的 timeout 內進入 `HOLD_STALE_ODOMETRY` |
| 9 | trajectory validity 過期 | 在設定的 timeout 內進入 `HOLD_STALE_TRAJECTORY` |
| 10 | validity flag 無效 | HOLD／拒絕，並提供 validity 無效原因 |
| 11 | odometry frame 錯誤 | `HOLD_INVALID_FRAME` |
| 12 | odometry 含非有限值 | `HOLD_INVALID_COMMAND`，並附上 state diagnostic |
| 13 | 時間倒退 | `HOLD_TIME_JUMP`；清除歷史資料並要求重新同步 |
| 14 | command 速度飽和 | command 有效且有界，並標示速度飽和 |
| 15 | command 加速度飽和 | command 有效且有界，並標示加速度飽和 |
| 16 | tracking 誤差過大 | `HOLD_TRACKING_ERROR` |
| 17 | 成功抵達目標並穩定 | 完成整段穩定時間後進入 `GOAL_HOLD` |
| 18 | 未抵達終點 | `TERMINAL_NOT_REACHED` 並進入 HOLD |
| 19 | yaw 跨越 wrap 邊界 | 使用最短 wrap 方向回授，且有界成功 |
| 20 | 拒絕無效 command | 獨立 validator 拒絕並選擇 `HOLD_INVALID_COMMAND` |

## 必要的 pure coverage

- sampler 的精確端點、區間前／內／後行為、每個插值欄位、未 wrap yaw 插值、無效 timestamps
  與非有限資料
- NED 中 feedback 符號與 feedforward 組合
- 各項限制的個別與合併順序套用、飽和回報、HOLD 建立、無效 command 拒絕及 validator 獨立性
- 輸入缺漏／false／過期、frame 錯誤、state 非有限、誤差過大、時間倒退或相等，以及時間跳變後重新同步
- 終點穩定、容差中斷／重設與終點 timeout
- 固定步長 plant 的確定性、一階響應、加速度限制、擾動，以及預設停用的確定性雜訊
- 獨立的 RMSE／最大值／頻率／飽和／HOLD／過期／穩定／完成度 metrics

## 必要的 ROS graph coverage

直接 graph 只包含固定 trajectory publisher、follower、offline plant 和有限 monitor。它會驗證
四個 output topics、精確 frames、candidate Twist semantics、連續有界 commands、state progression、
metrics，以及不存在 `/fmu/in/*`。Safety graph fixtures 至少涵蓋過期 odometry 與無效
trajectory。完整 graph 加入固定場景、A*、選定的 B-spline 或 fallback path、Phase 4
parameterizer、follower、plant 和 monitor，最後必須進入 `GOAL_HOLD`。

必要的 wrapper gates：

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
```

既有 Phase 2–4 的 unit、launch、wrapper、interface、import 和 legacy hash regressions 仍是
必要項目。產生的 outputs、logs、caches 和 model artifacts 不納入版本控制。
