# A* 確定性回歸規約

## 目的

這些測試固定安全性與外部可觀察的重要行為，不依賴脆弱的完整 grid-cell
序列。除非特別標示為離線 ROS 整合 fixture，否則所有 fixture 都是純 Python。

## 座標 fixture

- Origin and `+X/+Y/+Z` basis mapping.
- Exact inverse and fixed-seed finite-point round trips.
- Nonzero translation offsets on positions only.
- Velocity/acceleration vectors never receive translation.
- Planar heading and yaw conversions use explicit conventions.
- Non-finite values fail; quaternion conversion remains unsupported.

## 安全範圍 fixture

- Planning radius equals obstacle radius plus 0.18 m physical radius plus
  0.13 m static margin by default.
- Validation radius adds exactly 0.07 m.
- Point and segment clearance signs are checked at outside, tangent, and inside
  positions.
- Grid quantization reserve does not change either physical formula.
- Overflight tests cover clearly short, exact threshold, slightly tall,
  disabled mode, and negative/non-finite height rejection.

## Planner 測試案例

1. 沒有障礙物時，應產生安全的直接最終路徑。
2. 單一直接阻擋物會使路徑繞過 validation envelope。
3. 寬度大於兩倍 validation radius 的通道仍可通行。
4. 較窄的通道會被拒絕或繞行，不得穿越。
5. 起點靠近障礙物但位於 validation envelope 外時，仍視為有效。
6. 起點位於 planning/validation 禁止區域內時，應以結構化錯誤結束。
7. 目標位於禁止區域時，應以結構化錯誤結束。
8. 矮障礙物會被篩除，並保留直接路徑。
9. 高障礙物會保留並由路徑避開。
10. 完全阻擋且位於明確允許範圍內的障礙牆，結果應為 `no path`。
11. 成功時，raw、simplified 和 final paths 都必須保留精確端點。
12. 不安全的 RDP shortcut 或 simplification candidate 會被拒絕，並改採安全 fallback。
13. 提供經驗證的 raw path 時，fallback selector 會接受該路徑。
14. 輸入和設定完全相同時，重複執行必須得到相同結果。

斷言涵蓋成功/失敗、端點相等、連續淨空距離、路徑側向/路線特性、路徑長度上限、
確定性結果、fallback 原因與結構化診斷。只有重複執行相同輸入時，才會斷言完整
grid 序列完全相同。

## 驗證與 metrics 規約

- 每個輸入點都必須是有限值，且路徑至少包含兩個不同的端點。
- 預期起點和終點必須在設定的數值容差內完全相符。
- 每一條線段都必須依所有 validation radii 逐一檢查。
- 發生碰撞時，錯誤訊息必須指出線段索引和障礙物。
- 同時驗證選用的 planning bounds 與 waypoint 最大間距。
- Metrics 包含點數、2D 路徑長度、實際障礙物淨空距離、線段長度平均值／最大值，以及絕對
  heading 變化的平均值／最大值／變異數。
- Geometric metrics 不得標示為飛行平順度、加速度、jerk 或 tracking performance。

## ROS 離線測試案例

離線 harness 發布固定的 `isaac_world` 障礙物/起點/終點場景，等待 raw、simplified、
final paths 與 success status，接著驗證：

- 收到三種 paths；
- final frame 為 `px4_ned`；
- 轉換後的起點和終點精確保留；
- continuous validation 通過；
- 不存在 `/fmu/in/*` topic。

此流程不會啟動 Isaac、Pegasus、PX4、XRCE-DDS、camera、recorder 或 controller
process。Harness 結束後，launch timeout 可能停止持續執行的 planner；應將此情況
回報為預期 timeout，而不是測試失敗。
