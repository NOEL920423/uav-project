# Coordinate frame 與 clock 規約

## 狀態

Phase 2 完成第一個 planner 里程碑決策，並只在一個 pure module 中實作已驗證的位置與
向量轉換。此階段不建立 TF tree，也不宣稱已定義完整的 vehicle orientation 規約。

## 穩定的 frame 名稱

| Frame | 目標架構中的意義 | Phase 2 規則 |
|---|---|---|
| `isaac_world` | Isaac/Pegasus 原始場景座標、pose、障礙物、起點與終點 | 未經證明前不可稱為 ENU。原始場景訊息使用此 frame。 |
| `map` | 為未來符合 TF 的架構保留的 ROS planning/world frame | Phase 2 runtime TF 和 planner 訊息都不使用此 frame。 |
| `px4_ned` | 第一個整合里程碑使用的 PX4 local North-East-Down 位置/速度慣例 | Phase 2 obstacle、start、goal、path 的標準 planner frame。 |
| `base_link` | 位於所選 vehicle reference point 的 UAV body frame | Body 軸慣例與精確 prim/link 原點仍須驗證。 |
| `uav_fpv_camera` | 前向 camera 的 optical frame | 從 `base_link` 出發的 extrinsic transform 與 optical 軸慣例仍須校正。 |
| `uav_observer_camera` | observer camera 的 optical frame | 每個 episode/configuration 都必須明確指定此 camera 固定於 world 或連接於機體。 |

空的 `frame_id` 對空間訊息無效。若訊息 frame 與 consumer 設定的規約不同，consumer
必須拒絕該訊息，除非存在明確且 timestamp 有效的 TF transform。

## 沿用 Phase 0 的暫定位置轉換

第一個里程碑沿用實測使用的位置轉換：

```text
Isaac [x, y, z] -> PX4 local NED [north, east, down] = [y, x, -z]
PX4 local NED [north, east, down] -> Isaac [x, y, z] = [east, north, -down]
```

這是本專案使用的位置轉換，並不能證明它是通用的 ENU-to-NED pose transform。此轉換
沒有完整定義 quaternion handedness、yaw 零點、camera optical 軸、vehicle-body 軸、
local-origin offset、reset 行為，或 world origin 是否會在 episodes 之間移動。

Phase 2 的第一個 planning 決策採用 `px4_ned`。原始場景幾何維持在 `isaac_world`，並且
只在 ROS planner node 邊界轉換一次。Translation 參數 `ned_offset_x/y/z` 只套用於位置。
速度與加速度向量使用軸向轉換，不加 offset。只有在部署原點、orientation 和 TF 需求
完成驗證後，`map` 才能成為更高層的標準 frame。

平面 heading vectors 也採用相同的 XY 交換。依照 Isaac yaw 從 Isaac `+X` 逆時針起算、
而 NED yaw 從 north 朝 east 順時針起算的明確慣例，`yaw_ned = pi/2 - yaw_isaac`，並
正規化至 `[-pi, pi)`。這只是平面數學規約，不是完整的 body pose 定義。

目前不支援 quaternion 轉換、body 軸原點、camera optical 軸、原點 reset 與 covariance
語意。Phase 2 的 `nav_msgs/Path` 使用 identity pose orientation，Phase 2 consumer 不得
將其解讀為 heading。

**飛行整合前必須決策：**使用實際 Pegasus/PX4 驗證 quaternion/body/camera transforms、
原點 reset 與 covariance 行為。

## Clock 規約

所有目標 ROS nodes 都使用 `node.get_clock()` 的 ROS time 作為 message header stamp。在
simulation 中，所有參與的 node 必須設定 `use_sim_time=true`，且必須只有一個權威的
`/clock` publisher。`/clock` 必須存在、持續前進，並由 scene、image、vehicle-state、
command、planning 與 episode 訊息共同使用；否則 graph 不得開始 episode。

非 simulation 測試中，所有 node 都必須設定 `use_sim_time=false`，以 ROS system time 作為
共同 header 時間基準。混用 `use_sim_time` 設定屬於 configuration error。零 timestamp、
沒有 lifecycle reset 的時間倒退，或過期 timestamp 都會使相關 sample/command 遭拒絕。

Wall time 和 monotonic time 可以保留作為 recorder diagnostics 和 timeout 實作細節，但不
得作為跨 topic 同步的主要依據。Dataset joins 使用 `episode_id` 與 ROS header stamp；
image 和 CameraInfo 共用同一個 stamp，而 candidate/selected commands 則使用同一 ROS
time domain 中的 `TwistStamped.header.stamp` 檢查新鮮度。

沒有 header 的 `std_msgs/String`、`std_msgs/Bool`、services 和 action fields 不會自動
帶有 timestamp。其關聯與時間規則必須依照 interface contract，使用相關 stamped message
或 action lifecycle。

Phase 2 planner 由輸入/事件驅動，不執行有狀態的 timestamp 算術，因此不需要
`time_utils.py`；pure tests 也不依賴 ROS time 或 `/clock`。
