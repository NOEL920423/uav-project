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

## ROS 2 開發

```bash
./uav status
./uav doctor
./uav build
./uav test
./uav verify
./uav ml-test
```

`build` 使用 `colcon build --symlink-install`。`verify` 包含 build、套件測試、interfaces、
imports 與控制輸出邊界檢查。ROS 指令會以乾淨環境 source `/opt/ros/jazzy/setup.bash`
及本專案 `ros2_ws/install/setup.bash`，不 source 舊 workspace。
首次建置 `px4_msgs` 時可使用既有外部 source：

```bash
./uav build --base-paths src /home/noel_614420090/uav_ros2_ws/src/px4_msgs
```

`./uav shell` 開啟已 source 的子 shell，離開用 Ctrl+D。
`./uav interfaces` 列出正式 messages、services 與 topics。
`./uav topics`、`doctor`、`verify` 的非飛行檢查要求 graph 沒有 `/fmu/in/*`；請在飛行流程停止後執行。

## 診斷與資料管理

- `./uav bc-eval --diagnosis --episodes 1 --checkpoint <path> --seed <seed>`：單回合飛行診斷；需刻意執行。
- `python scripts/diagnostics/summarize_bc_startup.py <run>`：唯讀分析保存的 logs。
- `./uav expert-runtime-smoke`：persistent runtime reset 的獨立整合檢查。
- `./uav expert-scene-preview --seed <seed>`：僅預覽正式場景。
- `./uav px4-sitl-doctor`：檢查已執行的 SITL/XRCE readiness。
- `./uav px4-sitl-flight-check`／`isaac-runtime-flight-check`：對已執行的 runtime 發出受保護的自動 expert 飛行。
- `./uav artifact-publish`、`expert-collect-publish`、`backup`：既有 NAS 工具；本次精簡不執行上傳或刪除研究資料。

獨立離線 graph、synthetic publishers、搖桿與舊版 ML 指令已退役。
安全性、mux、gate、streamer、座標與時序的 regression tests 仍由 `./uav test` 執行。
