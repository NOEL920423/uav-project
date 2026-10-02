# 閉循環控制問題紀錄

## 目前狀態

控制問題分成兩類：影像／動作新鮮度，以及 PX4 控制邊界。兩類都不能靠放寬延遲門檻處理。本次程式修改保留既有 250 ms 動作期限與 PX4 gate 門檻；尚未用新的 Isaac Sim 飛行回合驗證修改效果。

## 第一類：影像與動作新鮮度

### 已確認的問題

檢視 `artifacts/evaluations/bc_flight/run_20260930T1749` 的 20 回合結果：10 回合被判為碰撞、10 回合為執行失敗；執行失敗中 8 回合是控制證據過期或無效，另 2 回合是 PX4 failsafe。碰撞是幾何碰撞分類，這筆資料沒有確認實際物理接觸。

BC 節點原本在 timer callback 內同步等待推論子程序回覆。失敗回合可看到 `_tick()` 最長約 170–334 ms 沒有再次執行；部分回合大部分時間花在等待推論子程序，CPU 執行時間很少。Episode 5 的新影像率約 4.3 Hz、影像接收間隔 P95 約 269 ms、指令所依據影像年齡 P95 約 352 ms，並記錄到 90 次舊動作重送、25 次新動作。這證明 callback 被推論等待卡住及舊動作被重送；但舊紀錄尚不能分辨影像解碼、模型各階段或程序通訊各花多少時間。

### 本次修改

- 推論 I/O 移到背景單工執行緒；等待中的推論請求只有一格，新影像會取代尚未開始的舊請求，不形成舊影像佇列。
- 動作保留對應的影像序號、影像來源時間、影像接收時間與推論起訖時間。每個推論動作只送出一次，不再用更新的訊息時間戳重送同一動作。
- 若影像接收至推論完成已超過既有 250 ms 動作期限，該動作不發布；沒有新鮮動作時，policy 回報未就緒，沿既有 supervisor／mux 路徑進入 HOLD。沒有重設時間戳，也沒有提高期限。
- timing 開啟時，另記錄請求編碼與 pipe 寫入、worker 等待、輸入解碼、影像解碼／前處理、CPU/GPU 傳輸、encoder、policy forward、輸出傳回，以及實際使用的裝置名稱。CUDA 階段計時會同步裝置，只在診斷開啟時執行。

非同步只讓 ROS 控制節點繼續收影像、里程計和處理 timer，不會讓模型變快。若推論本身仍超過 250 ms，對應動作會被丟棄並 HOLD；下一步應根據新增的階段耗時找出真正慢的部分，再決定如何加速，不能假定換 GPU 一定有效。

## 第二類：PX4 控制邊界

### 已確認的 failure path

- **Episode 16：來源切換時資料快照不一致。** 原 gate 分別接收 selected command、無時間戳的 source 字串、mux status，timer 可能將不同週期的資料組合。記錄顯示 gate 在 mux 已進入 HOLD 時仍拿到前一個 BC 指令，於 20 ms 內以「candidate source disagrees with mux active source」鎖定故障；隨後 streamer 停止送 setpoint，稍後 PX4 failsafe 出現。這是程式資料競態的直接證據，failsafe 是後續事件。
- **Episode 15：把 pre-flight check 當成持續飛行條件。** Gate 在機體已 armed 且進入 Offboard 時，因 `pre_flight_checks_pass=false` 轉入 `WAITING_VEHICLE_STATE` 並鎖定；之後 streamer 停止，PX4 failsafe 出現。PX4 `VehicleStatus` 對此欄位的定義是「所有解鎖所需檢查是否通過」，不是飛行中持續有效的健康旗標。其他運行健康條件仍由 failsafe、Offboard 狀態與本地位置／速度遙測檢查。

### 本次修改與限制

- Gate 只配對 header 時間戳完全相同的 selected command 與 mux status，candidate source 直接取該筆 mux status 的 active source。source 字串 topic 不再參與拼接。短暫不同步時沿用上一筆完整配對，直到原有 command timeout 使它失效；不把混合快照判成 source mismatch。
- Pre-flight checks 仍是解鎖前必要條件；機體 armed 後不再單靠該欄位撤銷 gate。failsafe、vehicle state、遙測有效性及其他 gate 檢查不變。
- 這兩項只處理已從 log 定位的 gate failure path；程式修改尚未經新的 SITL 回合驗證。Episode 15/16 的 PX4 failsafe 是否會因修改消失仍待重跑確認；ULog 也沒有證明其他回合中的 PX4 setpoint 追隨／控制律問題已排除。

## 閉循環測試目前保存的資料

每回合的原始程序與結果資料：

- `ready.log`、`scene.log`、`flight.log`、`isaac.log`、`xrce.log`：runtime readiness、場景準備、ROS／飛行節點、Isaac Sim 與 XRCE 的 stdout/stderr；失敗原因仍會保留。
- `result.json`：成功／失敗原因、步數、距離、路徑長度、終點位置、最小障礙物距離、checkpoint 身分等結果摘要。
- `trajectory_trace.json`：BC 動作對應的位置、姿態、目標距離、障礙距離與指令，用於回放及畫軌跡圖。
- `timing/*.jsonl`：各 ROS node 的 callback wall/CPU 時間、訊息接收時間及來源 stamp、timer 間隔、發布事件、輸入引用和 gate／streamer 狀態；scheduler 詳細資料只有在另外設定 `UAV_TIMING_SCHED_SECONDS` 時才會收集。新增的推論階段時間也寫在 BC node timing 檔。
- `timing_summary.md/json`、`control_summary.md`：從 timing、flight log、result 和 ULog 產生的分析摘要。
- `px4_ulog/`：診斷模式會啟動 PX4 ULog logger、複製 `.ulg` 並產生 manifest；摘要分析可另產生 topic CSV。ULog 時鐘未和主機校準，不能直接推算跨時鐘延遲。
- `policy_input_frames/` 暫存影像，以及 run 根目錄 `videos/*.mp4`、`videos/*.json`：每次成功推論的模型輸入影像和動作標註。
- run 根目錄 `summary.json`、`closed_loop_control_summary.md` 與結果圖：跨回合成功／失敗及結果趨勢。

## 停用額外診斷紀錄

在 `uav_ml/tools/bc_flight_evaluation.py` 修改檔案頂端的 `ENABLE_BC_FLIGHT_DIAGNOSTICS = True`：

- `True`：保留目前的 timing JSONL、ULog、推論輸入影片、逐步軌跡，以及由這些資料產生的 timing/control summaries。
- `False`：停止收集上述額外診斷資料；不要求 ffmpeg、不啟動 ULog capture、不建立 timing 報告、不寫 policy-input 影片或 trajectory trace。
- 不論開關為何，都保留程序 stdout/stderr log、`result.json`、run-level `summary.json` 和結果圖；這些是判讀回合結果與原始錯誤所需的基本資料。

此開關只控制 managed `bc-eval`／trace replay 流程的額外診斷資料，不改變新鮮度保護、HOLD 行為或飛行控制參數。預設維持 `True`，方便下一次針對本次修正取得證據；問題確認解決後可改成 `False`。
