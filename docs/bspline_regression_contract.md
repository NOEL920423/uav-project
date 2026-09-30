# B-spline 確定性回歸規約

## Pure basis 與 evaluation

- Open-uniform knot vectors 必須是有限且非遞減的，並在兩個 clamped endpoints 有正確的
  multiplicity。
- Basis values 在封閉 parameter interval 上構成 partition of unity。
- De Boor evaluation 必須回傳完全相同的第一個與最後一個 controls。
- 使用相同 inputs/configuration 重複呼叫時，必須得到相同結果。
- 無效 degree、knots、controls 或非有限中間值必須產生結構化診斷，不得流入 ROS publication。

## 短路徑與端點保留

- 少於兩個不重複 controls 時必須失敗。
- 兩個 controls 使用 degree 1；三個最多使用 degree 2；四個以上使用
  `min(configured degree, point_count - 1)`。
- 相鄰重複點必須以確定性方式移除；近乎重合的 controls 使用 planner numerical tolerance。
- Candidate 起點和終點必須在嚴格 tolerance 內等於已驗證的 A* endpoints，而且每個 point
  都保留固定的 `px4_ned` 高度。

## 空間重取樣

- 必須先進行 provisional evaluation，再依 cumulative arc-length interpolation 重取樣。
- Output spacing 應大致均勻，且不得超過設定上限加 numerical tolerance。
- Cumulative progress 必須有限且單調，endpoints 必須精確，最終 sample count 必須符合設定範圍。
- 零長度 input/provisional segments 必須明確移除或拒絕；最大 sample count 不足時拒絕 candidate。

## 獨立驗證

- 開放空間中的 curves 必須通過。
- 穿越障礙物或低於 clearance 的 curves 必須在連續 segment checks 中失敗。
- 非有限值、endpoint、altitude、bounds、spacing、零長度、curvature 和 self-intersection
  失敗都必須有確定性的原因。
- Validation radius 不得弱於 Phase 2 的 physical、static 與 continuous-clearance 範圍。
- Self-intersection 必須指出兩個不相鄰 segment indices；相鄰 segments 共用 endpoint 不算失敗。

## 選擇與 fallback

- 停用 B-spline 時選擇 `ASTAR_SIMPLIFIED`，並回報 `disabled`。
- 有效 candidate 選擇 `BSPLINE`，並將 valid/selected 設為 true。
- Candidate 被拒絕時，選擇已驗證的 baseline 作為 `ASTAR_FALLBACK`，保留整體成功狀態並
  提供拒絕原因。
- 若沒有有效的 A* baseline，整體結果必須失敗，且不得產生非空 final path。
- Candidate 被拒絕時不得再觸發一次 A* search。

## 幾何 metrics

- 直線的 curvature 應接近零。
- 固定圓弧應近似其已知的半徑倒數 curvature。
- Length、point count、physical clearance、segment length、heading-change 與 curvature 的
  mean/maximum/variance 都必須具確定性。
- 報告標示為 `Geometric Path Comparison`；不得宣稱 tracking、flight-time、collision-rate、
  acceleration、jerk 或 dynamic feasibility。

## ROS 離線 fixtures

有限 harness 接受具名的確定性 fixture。若 candidate state、source、frame、endpoint、final
validation 不符預期，或出現禁止的 topic，就會以非零狀態失敗。可選 fixture 名稱如下：

- `bspline-safe-open-space`
- `bspline-safe-single-obstacle`
- `bspline-rejected-corner-cut`
- `bspline-rejected-clearance`
- `bspline-disabled`
- `short-two-point-path`
- `three-point-path`
- `duplicate-control-point-path`
- `self-intersection-candidate`
- `curvature-limit-rejection`

Unit tests 負責直接的 control-point 邊界情況。ROS fixtures 負責驗證 publication、status、
fallback、frame、endpoint、連續 final-validation，以及 `/fmu/in/*` graph assertions。任何
隨機有限輸入測試都必須使用固定 seeds。
