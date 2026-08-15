# Phase 9.5 project organization

## Scope and baseline

This cleanup starts from Phase 9 commit
`352b50d460a958da8642331c0f3cbced81fd9841` on
`refactor/project-organization`. It changes repository organization and path
references only. It does not change navigation, control, PX4 safety, BC, PPO,
autoencoder, dataset, evaluation, or flight behavior.

No legacy file was deleted or deduplicated. No ignored dataset, checkpoint,
training run, runtime log, video, rosbag, pose log, or model output was moved.

## Previous and current layout

Previously, the repository root mixed direct Isaac/PX4 prototypes, wrappers,
ML utilities, USD scenes, and supported entry points. `ros2_isaac_scripts/`
mixed two active Phase 9 scripts with eight superseded episode/camera/planner
scripts, while the generic `scripts/` directory contained only ML workflows.

The organized tracked layout is:

```text
uav-project/
├── uav
├── train_ppo.sh
├── ros2_ws/src/                   # active ROS 2 packages and package tests
├── isaac/runtime/                # active Phase 9 Isaac/Pegasus runtime
├── uav_ml/                       # ML package
├── scripts/ml/                   # standalone ML/IsaacLab workflows
├── tools/isaac/                  # standalone Isaac diagnostic
├── tests/
│   ├── runtime/
│   └── ml/
├── legacy/
│   ├── isaac_direct_pipeline/
│   ├── isaac_ros2_episode_pipeline/
│   ├── pipeline/
│   ├── wrappers/
│   └── docs/
├── assets/usd/legacy_or_unclassified/
├── artifacts/README.md           # future plan only
└── docs/
```

## Moves

Active Isaac runtime:

- `ros2_isaac_scripts/7.isaac_uav_bootstrap.py` to
  `isaac/runtime/bootstrap.py`;
- `ros2_isaac_scripts/8.isaac_runtime_bridge.py` to
  `isaac/runtime/runtime_bridge.py`;
- `tests/test_isaac_runtime_contract.py` to
  `tests/runtime/test_isaac_runtime_contract.py`.

ML scripts, with source contents unchanged:

- the four previous `scripts/*.py` entry points to `scripts/ml/`;
- root `encode_bc_video.py` to `scripts/ml/encode_bc_video.py`.

Legacy code retained without deduplication:

- ten direct Isaac/PX4 and early demonstration scripts to
  `legacy/isaac_direct_pipeline/`;
- eight ROS-in-Isaac episode scripts to
  `legacy/isaac_ros2_episode_pipeline/`;
- `launch_uav_pipeline.py`, `uav_pipeline.sh`, and
  `README_BC_PIPELINE.md` to `legacy/pipeline/`;
- the three small `run_*.py` wrappers to `legacy/wrappers/`;
- the historical `docs/current_pipeline.md` to
  `legacy/docs/current_pipeline.md`.

Utility:

- `test_isaac_vscode.py` to `tools/isaac/test_connection.py`.

Unclassified tracked assets:

- all six root USD crates to `assets/usd/legacy_or_unclassified/`, retaining
  their original filenames and bytes.

## Reference updates

The active bootstrap now resolves `runtime_bridge.py` relative to its own
`__file__`. The runtime contract test, Phase 9 reproduction documentation,
and `uav` ML dispatch paths use the new locations. Legacy dynamic loaders and
wrappers use their new sibling paths, but their obsolete launch files,
services, and behavior were deliberately not repaired.

Current and historical documentation now distinguishes active runtime from
legacy source locations. `README.md`, directory READMEs, and the reserved
`artifacts/README.md` make each code and data category explicit.

## Regression results

Commands executed after the moves:

```bash
./uav test
./uav verify
./uav ml-test
pytest -q tests/runtime/test_isaac_runtime_contract.py
bash tests/test_uav_wrapper.sh
git diff --check
```

Results:

- ROS 2: 7 packages, 281 tests, zero errors/failures/skips;
- full verify: build, 281 tests, interfaces, import boundaries, generated-file
  checks, and `/fmu/in/*` safety checks passed;
- ML: 14 tests passed;
- Phase 9 embedded runtime contract: 4 tests passed;
- wrapper, shell syntax, Python compilation, and whitespace checks passed.

The ament flake8 runs retained their existing multithreaded `fork()` warning;
there were no test failures.

## Actual Phase 9 runtime verification

Isaac Sim was started with the reorganized entry point:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY \
  ./isaac-sim.streaming.sh --livestream 2 \
  --exec /home/noel_614420090/uav-project/isaac/runtime/bootstrap.py
```

The unchanged supported command was then run:

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=180 ./uav isaac-runtime-flight-check
```

Local ignored evidence is
`run_logs/px4-sitl-flight_20260815T124947Z.json`. It reports
`success=true`, terminal state `COMPLETE`, and:

- A*, B-spline, timed trajectory, and `ASTAR_EXPERT` evidence valid;
- PX4 setpoint rate 20.001 Hz, maximum gap 0.0502 s, zero dropped cycles,
  and no stream faults;
- OFFBOARD and armed confirmed;
- actual Isaac maximum altitude 1.561 m;
- actual Isaac tracking displacement 2.983 m;
- minimum actual Isaac goal distance 0.106 m;
- PX4 and Isaac landing confirmed, followed by disarm;
- final PX4 landed, disarmed, failsafe false, and streamer disabled.

This directly verifies that the active Isaac path relocation did not change or
break the Phase 9 flight behavior.

## Remaining risks

- The legacy pipeline remains intentionally unsupported and includes one
  known broken recorder wrapper plus references to removed launch/services.
- The six USD files remain unclassified; relocation proves byte preservation,
  not scene purpose or validity.
- Existing ignored artifacts remain in their historical top-level directories
  as requested. `artifacts/` is documentation for a future migration only.
- ML regression verifies software behavior, but Phase 9.5 did not launch
  training, collect data, or evaluate checkpoints.
- After the verified landing/disarm, headless Isaac Kit again ignored SIGTERM
  and required an exact-PID SIGKILL after PX4 had exited. This is an existing
  simulator process-shutdown risk, not a flight or repository-path failure.
