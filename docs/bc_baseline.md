# BC baseline training and evaluation

This is the formal first behavior-cloning reference for the canonical
high-rise expert dataset. It deliberately remains small and reproducible:

```text
FPV RGB / TOP RGB / FPV depth
  -> source-specific preprocessing to 3x72x128 float32 [0,1]
  -> frozen RgbAutoencoderV0
  -> latent64

latent64 + body-state8
  -> train-split mean/std normalization
  -> Linear(72,128), Tanh, Linear(128,128), Tanh,
     Linear(128,3), Tanh
  -> normalized [v_forward, v_right, yaw_rate]
```

The 8D state is body forward/right velocity, body-frame unit goal direction,
goal distance divided by 10 m and clipped to 1, and the previous normalized
three-axis action. Physical action limits remain 1.0 m/s forward, 0.8 m/s
right and 1.0 rad/s yaw. Only the selected image stream is used. The policy
receives no map, obstacle truth, full scene state, reward, or auxiliary target.

`LatentBcPolicy` is reused because it already implements the required
72D-to-3D MLP. The older class named `BcPolicyV0` has a depth/state input and
four-axis action contract, so using it here would violate the formal dataset
contract and create an incompatible parallel representation.

## Training

### Optional frozen ResNet18

AE remains the default; existing `ae-train`, `bc-train --encoder <path>`, and
AE checkpoints are preserved. Select ResNet18 with `--encoder-type resnet18`.
This uses ImageNet `IMAGENET1K_V1` weights, freezes all encoder parameters and
BatchNorm statistics, and trains only the existing BC MLP. No AE pretraining
or reconstruction loss is needed.

The encoder accepts RGB `[B,3,72,128]` in `[0,1]`, applies ImageNet mean/std
internally, and returns `[B,512]`. Concatenating the original state8 gives
`[B,520]`; BC still returns the same three normalized actions. The full
128x72 image is retained without center cropping. This is a deliberate
navigation-specific resolution choice, different from torchvision's default
224x224 classification crop. See the
[official ResNet18 weights documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Install torchvision in the Python environment used by `./uav`. For the current
project environment (`torch==2.12.0`), the matching version is:

```bash
python3 -m pip install --no-deps torchvision==0.27.0
```

Use a different compatible torchvision version if your PyTorch version differs.
AE does not require torchvision. The first ResNet run downloads the pretrained
weights into the normal Torch cache; inference loads the saved experiment weights
and does not download them again.

Train and then run one closed-loop test (replace the dataset name as needed):

```bash
./uav bc-train --dataset bc_expert_cube --image-source top \
  --encoder-type resnet18 --epochs 100 --batch-size 64 --learning-rate 0.001 --no-tensorboard \
  --output artifacts/experiments/bc/bc_expert_cube/top/resnet18/run_01

./uav bc-eval --image-source top_rgb --episodes 1 --visible \
  --checkpoint artifacts/experiments/bc/bc_expert_cube/top/resnet18/run_01/best.pt
```

Basic BC parameters are `--epochs`, `--batch-size`, and `--learning-rate`.
`--no-tensorboard` exits after training while still saving metrics and plots;
omit it to keep the managed TensorBoard server open until Ctrl+C.
`--encode-batch-size` controls encoder memory use separately (default 128).
Use a new output directory for another run; existing experiments are never
overwritten. Without `--output`, ResNet runs are automatically placed in
`artifacts/experiments/bc/<dataset>/<image_source>/resnet18/run_<timestamp>`;
the default AE latest pointer is unchanged. Pass the ResNet `best.pt` explicitly
to `bc-eval`; it reads the encoder choice from the checkpoint automatically.
Keep `resnet18_encoder.pt` in the original run location: the BC checkpoint
records its absolute path and SHA-256, as with existing AE checkpoints.

The live evaluator supports `top_rgb`, `fpv_rgb`, and `fpv_depth` checkpoints;
each subscribes only to its matching Isaac stream and applies the preprocessing
recorded in the checkpoint. ResNet training accepts `fpv_rgb` for offline
experiments, but not `fpv_depth`. Do not pass `--encoder` together with
`--encoder-type resnet18`.

ResNet runs produce the existing BC loss curves and action-error plots;
closed-loop evaluation produces the existing flight plots and result artifacts.
There are no reconstruction plots because the encoder is frozen.

### Existing AE training

If collection was interrupted after reaching its accepted-episode target, finish
validation and metadata offline before training (stop any collector first):

```bash
./uav expert-validate --dataset artifacts/datasets/bc_expert_cylinder_v2 --finalize
```

This revalidates every accepted episode and only then marks the collection
complete. It does not start Isaac Sim or collect additional episodes. Missing
episodes or invalid data still fail with the original validation error.

Run training explicitly after the dataset collection has completed and passed
its validator:

```bash
./uav bc-train --dataset bc_expert_cube --epochs 500
./uav bc-train --dataset bc_expert_cube --image-source fpv_rgb --epochs 100 \
  --encoder <fpv-ae-run>/best.pt
./uav bc-train --dataset bc_expert_cube --image-source fpv_depth --epochs 100 \
  --encoder <depth-ae-run>/best.pt
./uav bc-train --help
```

Defaults are the formal dataset at
`artifacts/datasets/bc_expert_cylinder_v1`, the existing frozen encoder at
`autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt`, Adam with learning rate
`1e-3`, batch size 64, at most 100 epochs, patience 12, and a fixed seed. TOP
is the default image source. Without `--encoder`, BC reads only the completed
matching AE provenance index for the same dataset/source and validates its
summary, checkpoint metadata, hash, preprocessing, architecture, and 64D
latent contract. It never falls back to another source or dataset.

FPV RGB is read from `samples.csv.image_path`. TOP RGB and FPV depth are joined
from `auxiliary.csv.observer_rgb_path` and `fpv_depth_path` by exact
`episode_id`/`sample_id` identity and primary timestamp. Availability, matched
status, timestamp error, and files are checked; there is no source fallback.
All three sources use the complete `fixed_global_top` comparison cohort and
therefore receive the same deterministic split for the same seed. Sparse
legacy TOP episodes are explicitly excluded from this comparison cohort.

The split is deterministic and episode-level, approximately 80/10/10. Frames
from one episode cannot cross train, validation, or test. The test split is
loaded only after the best-validation checkpoint has been selected. The exact
assignment and excluded records are saved in `split_manifest.json`.

Each run is written below:

```text
artifacts/experiments/bc_baseline/run_<UTC timestamp>/
  best.pt
  last.pt
  dataset_audit.json
  split_manifest.json
  training_config.json
  training_history.csv
  metrics.json
  summary.json
  tensorboard/events.out.tfevents.*
  plots/loss_curves.png
  plots/per_action_rmse.png
  plots/expert_vs_predicted.png
```

`best.pt` and `last.pt` include the model and optimizer states, normalization,
configuration, random seed, dataset manifest reference/hash, complete split,
encoder path/hash and observation/action contracts. The encoder is in eval
mode with gradients disabled throughout policy training. A successful run also
updates the ignored
`artifacts/experiments/bc_baseline/latest.json`, which evaluation uses by
default.

Offline metrics include equal-component normalized action MSE plus per-action
MSE, MAE, and RMSE. They answer: **can BC imitate held-out expert actions?**
They do not establish that the UAV can navigate successfully.

BC output is automatically placed under
`artifacts/experiments/bc/<dataset>/<source>/run_<timestamp>`. The CLI starts a
managed TensorBoard server by default and keeps it running after successful
training until Ctrl+C; `--no-tensorboard` disables only the server, not
event-file recording. TensorBoard records
`bc/train_action_loss` and
`bc/validation_action_loss` each epoch. It records
`bc/test_forward_rmse`, `bc/test_right_rmse`, and
`bc/test_yaw_rate_rmse` once after reloading `best.pt`. Start it with:

Retained events can later be reopened with
`tensorboard --logdir <run-directory>/tensorboard`.

For a remote server, use an SSH tunnel such as
`ssh -L 6006:localhost:6006 user@server`, then open
`http://localhost:6006` locally. The formal closed-loop runtime accepts the
TOP RGB checkpoint contract.

## Closed-loop evaluation

After training has produced `best.pt`, run:

```bash
./uav bc-eval --episodes 20
./uav bc-eval --help
```

An explicit checkpoint can be selected with `--checkpoint PATH`. The tool
loads the exact encoder recorded in the checkpoint, verifies its SHA-256,
reuses the training preprocessing and normalization, and launches the managed
Isaac Sim, Pegasus, PX4 SITL, and ROS 2 flight stack.

During every rollout, `CONTROL SOURCE = BC_POLICY`. The evaluator calls only
the BC policy for actions; it never calls the environment's A* expert, blends
an expert action, or adds privileged information to the policy observation.
A* remains internal only for generating reachable randomized environments and
is not a control source. Collision and out-of-bounds termination are reported
as collision failures.

Evaluation output is written below:

```text
artifacts/evaluations/bc_flight/run_<timestamp>/
  metrics.json
  episode_<index>/result.json
  plots/
```

Each episode records seed, success, collision, timeout, terminal reason,
minimum/final goal distance, flight duration, measured path length, policy
ownership, safety-abort state, and blending state. Aggregate metrics contain
counts/rates and mean distance, duration, and path length. These closed-loop
metrics answer: **can BC actually fly to the goal by itself?** A low offline
MSE must never be interpreted as navigation success.

## Scope and current risk

This baseline does not perform encoder fine-tuning, RGB-D fusion, recurrent or
attention modeling, DAgger, GAIL, PPO, or hyperparameter search. The closed-loop
environment uses deterministic fixed-height dynamics and an Isaac-rendered
camera; it is useful policy-only evidence but does not replace a separately
authorized PX4/Pegasus learned-policy flight or hardware validation. Domain
shift between the formal Pegasus dataset and this evaluation renderer remains
an explicit risk.

Generated datasets, checkpoints, metrics, and plots are ignored by Git.

## 閉循環時間紀錄與閱讀方式

`./uav bc-eval` 會在每回合的資料夾自動保存以下診斷資料，沿用既有
node 與 evaluator，不需另外啟動 Python 診斷程式：

- `timing_summary.md`：中文報告，先看回合結果與首次故障，再看各段
  平均、P95、最大耗時及事件順序。原始錯誤文字保留英文以方便搜尋。
- `timing_summary.json`：相同觀測數據的結構化版本。
- `control_summary.md`：每回合優先閱讀的中文控制摘要，只保留結果、新影像率、影像到 PX4、影像年齡、動作重送、最大 timer 間隔、ULog 可否比較與下一個假說。
- `closed_loop_control_summary.md`：整次 closed-loop 的回合總覽；先看這份，再依異常回合連結打開其 `control_summary.md`。
- `timing/*.jsonl`：各 node 的原始時間事件，可依報告提供的檔名與行號回查。
- `px4_ulog/`：ULog、保存清單及 logger 原始輸出。evaluator 會在
  Pegasus 清除 temporary rootfs 前持有檔案並保存；若保存失敗會明確報錯。

時間事件包含主機 monotonic、ROS 與 wall clock、來源訊息標記、發布與
callback 時間、timer 間隔、判斷時快取年齡、影像讀取與編碼、BC 推論耗時，
以及 gate／streamer 狀態與當時有效門檻。診斷不修改控制、安全門檻或
PX4 參數；ULog 使用 logger 指令啟停。

**發布到 callback 的耗時包含傳輸與排程，不能直接稱為 DDS 延遲。**
快取年齡也不代表傳輸耗時，且快照可能包含未選用的控制來源。
目前尚未量到影像真正擷取時間、DDS 內部排隊、PX4 收到命令後的執行耗時，
也未將 PX4 boot clock 與主機時間校準。紀錄本身會增加少量負載，
尚未透過對照實驗量化；因此單回合數值不能當成固定門檻的充分依據。

只重新整理已保存的資料、不啟動模擬：

```bash
python scripts/diagnostics/summarize_bc_startup.py artifacts/evaluations/bc_flight/<run_directory>
```

### FPV and action timing evidence

FPV `image_read` events now separate pose update, annotator read and JPEG
encoding. The paired `publish` event carries a publication sequence and ROS
message key. This sequence is **not** a renderer frame ID. Capture time,
render completion and whether the pose update applies to the returned frame
remain explicitly unknown; no extra render or sensor rate change is made.

BC records the image, odometry and goal source keys and host receipt times
used for each inference. Every action has an ID, an inference completion
time, and a repeated-action flag. Mux, gate and streamer publications retain
their consumed input references in the diagnostic JSONL only. ROS messages
and model inputs are unchanged. The report joins these references only for
BC-selected, safe-to-forward commands. It reports image receipt age at PX4
publication, including repeated actions, and the first publication per action.
These measurements stop at the host publisher API, not at PX4 reception.
Old recordings without references remain unlinked rather than guessed.

The report inspects saved ULogs using optional `pyulog` (`python -m pip install
pyulog` in the report interpreter). It exports the available setpoint,
position/velocity, attitude and timesync topics to CSV beside each ULog,
lists absent topics and preserves dropout timestamps/durations in JSON.
An import or parse failure is shown in stderr and the report. ULog timestamps
are not automatically host timestamps or measured actuation times; logging
may downsample setpoints. No host-to-PX4 delay is calculated without clock
calibration. CSV NaN values retain PX4's unset-field representation.
When timing is enabled, the streamer also records `/fmu/out/timesync_status`
offsets, remote timestamps, protocol and round-trip time together with host
clocks. Missing message support is reported. An absent publisher yields no
witnesses; the report does not treat timesync availability alone as validated
host/PX4 calibration or control execution latency.

### Bounded scheduling diagnostics

Control callbacks record entry, exit, wall duration and thread CPU duration,
with PID/TID. To additionally capture Linux scheduler and I/O counters for
the first 30 seconds of each node in an already planned targeted run:

```bash
UAV_TIMING_SCHED_SECONDS=30 ./uav bc-eval --episodes 1 --checkpoint <same_checkpoint> --seed <same_seed>
```

The environment variable is inherited by runtime processes;
the extra counter sampling stops automatically and is disabled by default.
Counters are sampled at timer entry, not by a new worker or subprocess.
`schedstat` measures the calling thread's runqueue wait. Block I/O delay is
reported in kernel ticks; process read/write bytes are supporting evidence,
not a duration. Zero counters do not establish that kernel accounting is
enabled. Missing `/proc` evidence is explicitly recorded and disables further
sampling. Callback wall time minus CPU time includes preemption and other
waits, so it must not be labeled I/O delay. Diagnostics have overhead that
has not yet been measured in a simulator comparison.

### Ground confirmation before startup

The supervisor treats missing or expired land detection as unknown, not airborne.
Startup requires fresh disarmed and landed evidence before selecting lifecycle
control. Unknown ground evidence pauses startup/recovery without enable, mode,
arming, or reset requests; an already selected source is switched to HOLD.
The existing readiness timeout, overall startup deadline, recovery limit, and
250 ms command timeout remain unchanged. Confirmed airborne evidence aborts;
OFFBOARD request/history continues to prohibit startup recovery. Status `landed`
is `null` while land evidence is missing or older than 1.5 seconds.

When timing is enabled, control and status publishers also record
`publish_api_start` and `publish_api_end`, including topic, source key, wall time,
thread CPU time, and success. Existing publication/lineage events are retained.
The API duration excludes diagnostic writes and does not measure PX4 reception
or execution. A long API duration identifies a wait inside the publish call,
but does not alone distinguish DDS locks/backpressure from CPU preemption.
Use the existing bounded scheduler counters above to investigate that distinction;
zero/unavailable accounting is not evidence of zero scheduling delay.

Targeted regressions (run manually in the project's test environment):

```bash
PYTHONPATH=ros2_ws/src/uav_px4_control:. python -m pytest ros2_ws/src/uav_px4_control/test/test_bc_startup_recovery.py ros2_ws/src/uav_px4_control/test/test_bc_flight.py ros2_ws/src/uav_px4_control/test/test_px4_output_gate.py
PYTHONPATH=ros2_ws/src/uav_px4_control:. python -m pytest tests/ml/test_bc_flight.py -k TimingReportTests
```
