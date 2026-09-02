# 這個 Repository 用來記錄我的碩論跟中正大學孫計畫的開發，所以內容會隨研究進度持續調整，能否畢業全看祂了，現在看來還遙遙無期 --2026/09/02



## Repository の 結構

- `ros2_ws/src/`：目前使用中的 ROS 2 套件與各套件測試。
- `isaac/runtime/`：Isaac Sim / Pegasus 的主要執行環境與 pose bridge。
- `uav_ml/`：BC、PPO、Autoencoder、Dataset、Inference 與 Evaluation 相關模組。
- `scripts/ml/`：獨立的 IsaacLab / ML 執行腳本。
- `tools/`：除錯、診斷與開發輔助工具。
- `tests/`：整個專案層級的 runtime 與 machine learning 測試。
- `legacy/`：過去開發過程中保留的舊版程式，目前正式飛行流程不會使用。
- `assets/usd/legacy_or_unclassified/`：舊版或尚未重新分類的 USD 場景與資源。
- `docs/`：系統設計、問題紀錄、研究里程碑與開發文件。
- `artifacts/`：實驗、測試與診斷產生的輸出結果。

未來會考慮簡化資料結構

```bash
./uav expert-collect --episodes 100 --dataset bc_expert_cube
./uav expert-collect --episodes 100 --dataset bc_expert_cube --resume
./uav expert-collect --help
```

Each completed invocation may append any positive episode count; there is no
fixed total target. It writes only to
`artifacts/datasets/bc_expert_highrise_v1/`. See
[`docs/expert_dataset_collection.md`](docs/expert_dataset_collection.md) for
the frozen scene/camera/data contracts, progress, resume, QA, and validation.

## 目前專案主要包含：

- Isaac Sim 無人機模擬環境
- Pegasus Simulator 整合
- PX4 OFFBOARD 飛行控制
- A* path planning
- FPV 與 observer camera
- Camera image recording
- ROS 2 pose 與飛行資料記錄
- Manual / joystick control 實驗
- Expert Dataset 自動蒐集
- Autoencoder 與 Behavior Cloning
- Closed-loop flight evaluation

## Notes

Dataset、rosbag、log、image、video、checkpoint 與其他實驗輸出不會丟到 github 上，畢竟現在是一坨 ![alt text](image.png) 。

