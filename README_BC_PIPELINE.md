# ROS 2 UAV 影像與 BC 訓練操作手冊

## 1. 從 Windows PowerShell 登入

先連上中正大學 VPN，再執行：

```powershell
ssh noel
```

若 SSH alias 不可用：

```powershell
ssh noel_614420090@140.123.122.115
```

以下命令是在 SSH 登入後的 Server shell 執行。

## 2. 啟動完整飛行系統

```bash
~/uav-project/uav_pipeline.sh start-flight
```

這個命令會：

1. 啟動 ROS 2 lookahead follower 與 mission orchestrator。
2. 以 headless WebRTC 模式啟動 Isaac Sim。
3. 自動載入 Default Environment。
4. 自動建立 Iris、Pegasus PX4 backend、相機與 ROS 2 episode manager。
5. 自動開始 Isaac timeline，不需要手動按 Play。

查看狀態：

```bash
~/uav-project/uav_pipeline.sh status
```

## 3. 跑一個 episode

```bash
~/uav-project/uav_pipeline.sh run
~/uav-project/uav_pipeline.sh wait
```

影像只會在 follower 回報 `ACTIVE`，也就是 PX4 已確認 OFFBOARD 與 ARMED 後開始；進入 LANDING 時會先關閉 FPV/TOP 與 pose 檔案。

要跑下一個獨立 episode，先重置 Isaac/PX4：

```bash
~/uav-project/uav_pipeline.sh restart-isaac
~/uav-project/uav_pipeline.sh run
~/uav-project/uav_pipeline.sh wait
```

`restart-isaac` 只允許在 IDLE、COMPLETE 或 FAILED 狀態執行，不會在飛行途中直接關閉。

## 4. Isaac Sim / WebRTC 要做的事

使用自動 bootstrap 時，不需要在 Isaac Sim 手動：

- 載入場景
- Spawn Iris
- 啟動 PX4 backend
- 按 Play
- 執行 Script Editor 腳本

WebRTC 僅用來監看畫面，可連可不連，不影響 off-screen FPV/TOP 記錄。不要再手動 Spawn 第二台 Iris 或重複按 Play。

## 5. 停止 Isaac 並跑 BC 訓練

訓練前關閉 Isaac/PX4，釋放 GPU：

```bash
~/uav-project/uav_pipeline.sh stop-isaac
~/uav-project/uav_pipeline.sh train
```

查看訓練狀態：

```bash
~/uav-project/uav_pipeline.sh train-status
```

或直接監看 ROS 2 topic：

```bash
source /opt/ros/jazzy/setup.bash
source ~/uav_ros2_ws/install/setup.bash
ros2 topic echo /uav_bc/training_status
```

列出資料與模型：

```bash
~/uav-project/uav_pipeline.sh list-data
```

## 6. ROS 2 原始服務命令

```bash
source /opt/ros/jazzy/setup.bash
source ~/uav_ros2_ws/install/setup.bash

ros2 service call /uav_mission/run_episode std_srvs/srv/Trigger '{}'
ros2 service call /uav_mission/get_status std_srvs/srv/Trigger '{}'
ros2 service call /uav_mission/abort std_srvs/srv/Trigger '{}'

ros2 service call /uav_bc/train std_srvs/srv/Trigger '{}'
ros2 service call /uav_bc/get_status std_srvs/srv/Trigger '{}'
ros2 service call /uav_bc/cancel std_srvs/srv/Trigger '{}'
```

## 7. 資料與模型位置

```text
影像：~/uav-project/uav_vision_dataset/dual_camera_episode_bc_astar_*/
Pose：~/uav-project/ros2_uav_pose_logs/uav_pose_bc_astar_*.csv
模型：~/uav-project/uav_bc_models/bc_*/best.pt
訓練紀錄：~/uav-project/uav_bc_models/logs/
```

目前的 BC target 是從專家飛行軌跡推導出的未來 0.5 秒 Isaac XYZ 速度。模型輸出尚未直接接到 PX4；必須先做更多場景資料、closed-loop 模擬驗證與安全限制，才能考慮讓模型控制飛行。

