# 開發與驗證指令

在 VS Code Remote-SSH 的 Ubuntu Integrated Terminal，從 repository 根目錄使用 `./uav`。
收集至評估的操作見 [SystemUserManual.md](SystemUserManual.md)，元件見 [architecture.md](architecture.md)。

## 日常入口

| 指令 | 用途 |
|---|---|
| `./uav expert-collect` | 收集可續跑的 expert episodes |
| `./uav expert-validate` | 驗證既有 collection |
| `./uav ae-train` | 訓練 AE encoder |
| `./uav resnet-train` | 訓練 ResNet18 encoder |
| `./uav bc-train` | 訓練 BC 並做 held-out offline evaluation |
| `./uav bc-eval` 或 `./uav eval` | Isaac/Pegasus/PX4 閉環飛行評估 |
| `./uav bc-eval-plot <run>` | 使用既有結果重新產圖 |

每個入口均可使用 `--help`。Dataset、來源、split、seed、checkpoint 與場景由本次實驗指定；
精簡專案不改寫舊的 `config/profiles/current.json` 或已保存實驗設定。

## 建置與 preset

```bash
./uav build
./uav profile current expert-collect
./uav profile current ae-train
./uav profile current bc-train
./uav profile current bc-eval --episodes 20
```

`build` 使用 `colcon build --symlink-install`。ROS 指令會以乾淨環境 source `/opt/ros/jazzy/setup.bash`
及本專案 `ros2_ws/install/setup.bash`，不 source 舊 workspace。
首次建置 `px4_msgs` 時可使用既有外部 source：

```bash
./uav build --base-paths src /home/noel_614420090/uav_ros2_ws/src/px4_msgs
```

`profile` 只套用 `config/profiles/` 中既有的參數，額外 CLI 參數可覆蓋 preset。
內部的 runtime readiness、scene preparation、expert flight、BC flight 與 trace replay
入口由收集及閉環流程呼叫，不需手動執行。

## 開發測試

入口已精簡為收集、訓練、閉環測試及直接相關的輔助指令。
開發測試仍可從 Ubuntu Integrated Terminal 執行：

```bash
bash tests/test_uav_wrapper.sh
python3 -m unittest discover -s tests/ml -p 'test_*.py'
```

ROS 套件測試需使用 Jazzy 與本專案已建置的 overlay：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
cd ros2_ws
colcon test
colcon test-result --verbose
```

飛行診斷仍可使用 `./uav bc-eval --diagnosis`，保存結果的分析可使用
`python3 scripts/diagnostics/summarize_bc_startup.py <run>`。
備份與 NAS 工具保留在 `tools/backup.sh` 及 `uav_ml/tools/artifact_publish.py`。
