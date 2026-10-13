# AGENTS.md

## 1. Project Overview

本專案為 UAV Autonomous Navigation Research Project，主要包含：

- NVIDIA Isaac Sim / Pegasus Simulator
- PX4 Autopilot / SITL
- ROS 2
- Expert Data Collection
- Autoencoder / Behavior Cloning / PPO
- Offline and Closed-loop Evaluation

長期研究目標是在標準化測試條件下，達成 20/20 successful closed-loop flights。

此為 Research Target，不是每項 Codex 任務的完成標準。不得為達成目標而擅自調整測試條件、排除失敗結果或修改評估規則。

## 2. Core Principles

所有工作遵循以下優先順序：

**Correctness & Safety → Minimal Scope → Reusability → Readability → Efficiency**

1. Understand before modifying：修改前先理解直接相關的架構、資料流程與介面。
2. Minimal Scope：只處理本次任務相關的問題，不進行無關重構。
3. Reuse Existing Code：能合理沿用現有功能時，避免建立重複實作。
4. Readability：程式應易於研究者理解、修改及除錯。
5. Evidence-based Decisions：根據程式、Logs、Tests 和 Artifacts 判斷問題，不依靠猜測修改。
6. Preserve Existing Behavior：除非明確要求，不得改變既有 CLI、資料格式、模型契約或控制行為。

如果使用者明確要求 Read-only Audit、Diagnosis 或 No Modification，不得修改任何檔案。

## 3. Efficient Repository Inspection

- 優先閱讀與本次任務直接相關的檔案及其依賴。
- 已知 Failure Path 時，先沿該路徑調查。
- 不要無必要地掃描整個 Repository 或讀取所有 Python 檔案。
- 只有在局部資訊不足以判斷問題時，才擴大搜尋範圍。
- 優先使用既有 Logs、Artifacts、Diagnostics 和 Tests。
- 如果已有可靠證據排除某個假說，不要在缺少新證據時反覆調查。

Token Efficiency 不得凌駕於 Correctness 之上。

## 4. Code Architecture & Maintainability

以清楚的 Module Responsibility 和 Data Flow 為優先，不單純追求最少檔案數量。

- 小型且相關的功能，優先整合至現有模組。
- 如果既有模組過大、職責不同，或新增功能會增加耦合，可以建立獨立模組。
- 新增模組前，先評估是否能重用現有實作。
- 避免 Unnecessary Abstraction、Duplicated Logic 和 Circular Dependencies。
- 不要為小型功能過度新增 Wrapper、Adapter、Manager、Monitor、Fixture 或 State Machine。
- ROS2 Node 和核心邏輯可合理分離，但不要為簡單邏輯建立額外 Node。
- Diagnostic / Offline Tools 優先放在 `tools/` 或適當的診斷模組。
- Tests 應放在對應測試目錄，不混入主要 Runtime Logic。

新增 Python 檔案時，簡短說明新增的理由及其責任。

**Prefer cohesive modules over fewer files.**

## 5. Machine Learning Research Integrity

涉及 Dataset、Model、Training 或 Evaluation 的修改，必須特別注意 Research Reproducibility。

- 保留 Dataset Identity、Episode IDs、Random Seed、Hyperparameters 和 Checkpoint Information。
- 不得未經要求修改 Dataset Split、Action Labels、Normalization 或 Evaluation Conditions。
- 不得將 Training-set Evaluation 當作 Generalization Performance。
- 明確區分 Offline Prediction Metrics 與 Closed-loop Flight Metrics。
- Diagnostic Experiments 必須與 Production Training Artifacts 隔離。
- 不得意外覆寫現有 Models、Datasets 或重要 Checkpoints。
- 修改 Model Inputs、Outputs、Coordinate Frames、Action Units 或 Timing Contracts 時，必須檢查上下游相容性。
- 實驗失敗應保留原始結果，不得為提高 Success Rate 而隱藏或刪除失敗紀錄。

Research conclusions must be supported by reproducible experimental evidence.

## 6. Debugging & Error Handling

診斷時，明確區分：

- Confirmed：已有直接證據支持。
- Hypothesis：合理但尚未驗證。
- Unknown：目前證據不足。

優先根據原始 stderr、traceback、telemetry、ULog、process exit code 與 artifacts 進行分析。

- 不要將時間先後關係直接認定為因果關係。
- 如果多個假說都合理，優先設計最小的 Discriminating Test。
- 不要只根據症狀就大幅修改系統。
- 保留原始 ERROR、FATAL 和 Traceback。
- Summary Classification 不得取代原始 Failure Reason。
- 新增 Subprocess 或 Worker 時，必須避免 stdout/stderr PIPE 阻塞。
- 不要透過捕捉並忽略 Exception 來使測試表面通過。

## 7. Testing & Expensive Operations

根據 Change Impact 選擇必要的測試。

建議測試順序：

1. Static Inspection
2. Syntax / Import Check
3. Unit Tests
4. Targeted Integration Tests
5. Smoke Tests
6. Isaac Sim Runtime Tests
7. Full Experiments

- 優先執行低成本、與本次修改直接相關的測試。
- 測試通過且相關程式未改變時，不要無理由重複執行。
- 修改少量程式不代表一定要執行完整 Isaac Sim Regression。
- 如果 Targeted Tests 已提供足夠證據，可以停止擴大測試。
- 如果修改涉及 Flight Control、Timing、Model I/O 或 Runtime Integration，不能僅靠 Syntax Check 宣稱系統行為正確。
- 除非使用者要求，不要自行增加 Simulation Episodes 或執行完整昂貴實驗。

若必要測試因環境、資源或時間限制無法執行，應明確回報尚未驗證的部分。

## 8. Git & Data Safety

- 不得修改與本次任務無關的既有變更。
- 不得擅自使用 `git reset --hard`、`git clean -fd` 或其他 Destructive Operations。
- 不得自動 Commit、Push 或建立 Branch，除非使用者明確要求。
- 修改前後應保留使用者既有的 Uncommitted Changes。
- 不得擅自刪除 Dataset、Checkpoint、Logs 或實驗結果。

## 9. Code Quality & Learning Support

- Code、Variables、Functions、Classes、Comments 和 Docstrings 一律使用英文。
- 優先使用清楚、具有意義的 Naming。
- 保持 Core Algorithm、Data Processing 與 I/O Responsibilities 易於辨識。
- 避免不必要的間接呼叫與複雜抽象。
- 不需要為每一行程式加入註解。
- 當使用者要求學習或理解實作時，應指出 Core Functions、Inputs、Outputs 和主要演算法流程。
- 優先解釋實際研究相關的核心程式，而不是所有 Boilerplate。

**AI-assisted development should improve both research productivity and code understanding.**

## 10. Task Completion & Reporting

完成工作後，採用簡短的結構化回報：

### 修改內容
列出實際修改的檔案及主要原因。

### 測試結果
列出執行的測試、Pass/Fail，以及重要錯誤。

### 尚未解決
只列出真正尚未解決的問題或必要的後續驗證。沒有則省略。

### 簡單總結
以 2～5 點或幾句話說明目前狀態與下一步。

避免重複背景、無意義的冗長描述及大量不影響結果的細節。

若使用者要求 Audit、Research Analysis、Implementation Plan 或 Code Walkthrough，則依任務需要提供充分細節，不受一般簡短回報格式限制。

## 11. Task-specific Instructions

`AGENTS.md` 定義長期適用的 Project-wide Rules。

具體的 Experiment Configuration、Hyperparameters、Episode Selection、Evaluation Metrics 和 Task-specific Constraints，應由當次 Task Prompt 定義。

不得將單次實驗的特殊設定，自動套用至其他正式實驗。