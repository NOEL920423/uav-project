# Architecture Map

This map covers the formal workflow only:

```text
expert-collect -> ae-train -> bc-train -> bc-eval
```

## Formal training flow

```text
./uav expert-collect
  -> uav_ml.tools.expert_collect
  -> uav_ml.tools.persistent_runtime
  -> isaac/runtime/bootstrap.py
     (Isaac Sim + Pegasus + PX4 SITL + runtime_bridge)
  -> ./uav expert-run-episode
  -> px4_sitl_flight.launch.py
  -> expert_dataset_recorder
  -> artifacts/datasets/<dataset>

./uav ae-train
  -> uav_ml.train_autoencoder
  -> RGB episode dataset -> RgbAutoencoderV0 checkpoint

./uav bc-train
  -> uav_ml.tools.bc_baseline
  -> frozen AE + expert dataset -> LatentBcPolicy checkpoint
```

`ae-train` and `bc-train` are ROS-, Isaac-, and PX4-independent.

## Formal BC closed-loop flow

```text
./uav bc-eval
  -> uav_ml.tools.bc_flight_evaluation
  -> isaac/runtime/bootstrap.py
     (Isaac Sim + Pegasus + PX4 SITL + runtime_bridge)
  -> ./uav bc-flight-run
  -> bc_flight.launch.py

TOP RGB + PX4 odometry + scene goal
  -> bc_policy
  -> control_mux
  -> px4_mapping_gate
  -> px4_setpoint_streamer
  -> PX4 /fmu/in trajectory setpoint + offboard mode
  -> Pegasus vehicle in Isaac

bc_flight_supervisor + bc_episode_monitor
  -> lifecycle, termination, result JSON, traces, plots
```

`bc-eval` is the formal Isaac/Pegasus/PX4 path. `bc-eval-surrogate` is a separate software-surrogate evaluator; deprecated `bc-closed-loop` routes to that surrogate path.

## Main ROS 2 nodes

| Node | Formal use | Role |
|---|---|---|
| `scene_bridge` | collection, BC eval | Isaac pose/status -> validated scene obstacles, start, goal |
| `astar_planner` | collection | Scene -> A* path |
| `trajectory_parameterizer` | collection | Path -> timed trajectory |
| `trajectory_follower` | collection | Trajectory -> `ASTAR_EXPERT` command candidate |
| `expert_dataset_recorder` | collection | Synchronizes images, odometry, expert command and flight evidence into episodes |
| `px4_odometry_bridge` | collection, BC eval | PX4 vehicle odometry -> ROS odometry |
| `control_mux` | collection, BC eval | Selects exactly one command source |
| `px4_live_telemetry_adapter` | collection, BC eval | PX4 status/control/odometry/failsafe -> gate-facing telemetry |
| `px4_mapping_gate` | collection, BC eval | Fail-closed command validation and PX4 candidate mapping |
| `px4_setpoint_streamer` | collection, BC eval | Sole trajectory-setpoint/offboard-mode publisher to PX4 |
| `px4_vehicle_command_owner` | collection, BC eval | Sole vehicle-command publisher to PX4 |
| `px4_sitl_flight_supervisor`, `px4_sitl_flight_monitor` | collection | Expert mission lifecycle and evidence |
| `bc_policy` | BC eval | TOP RGB AE+BC inference -> BC command candidate |
| `bc_flight_supervisor` | BC eval | Arm/takeoff/enable/landing lifecycle |
| `bc_episode_monitor` | BC eval | Goal/collision/timeout decision and result artifact |

## Launch graphs

| Launch file | Used by | Nodes unique to the graph |
|---|---|---|
| `uav_px4_control/launch/px4_sitl_flight.launch.py` | `expert-collect` | A* planner, parameterizer, follower, expert recorder, expert supervisor/monitor |
| `uav_px4_control/launch/bc_flight.launch.py` | `bc-eval` | BC policy, BC supervisor, BC episode monitor; intentionally no planner/follower |

## Classification

### ACTIVE

- `uav`
- `uav_ml/tools/{expert_collect,persistent_runtime,bc_baseline,bc_flight_evaluation,training_cli}.py`
- `uav_ml/train_autoencoder.py`, `uav_ml/train_bc.py`
- Formal dataset, AE/BC model, and BC inference modules
- `isaac/runtime/{bootstrap,runtime_bridge,episode_scene,formal_expert_sensor_contract,scene_visual_materials}.py`
- Active nodes and launch/config files named above from `uav_navigation`, `uav_scene_bridge`, `uav_data_recorder`, and `uav_px4_control`

### SUPPORT

- Expert collection validation/finalization/visual-QA helpers
- BC flight plotting helper
- Formal navigation and PX4 configuration YAML files

### DIAGNOSTIC

- `scripts/diagnostics/summarize_bc_startup.py`
- Offline harnesses, comparison tools, smoke checks, and PX4 diagnostic launch paths

### LEGACY

- `legacy/**`
- `bc-closed-loop` command path
- `scripts/ml/evaluate_isaac_bc_closed_loop.py`

### UNKNOWN / outside the formal flow

- `uav_camera_bridge` and `uav_bringup/uav_system_scaffold.launch.py`: scaffold nodes, not launched by the formal graphs.
- `uav_ml/train_latent_bc.py`, `uav_ml/envs/**`, `scripts/ml/collect_isaac_bc_demonstrations.py`, `scripts/ml/train_isaac_bc_initialized_ppo.py`, and `scripts/ml/isaac_city_smoke.py`: alternate IsaacLab/latent/PPO workflow.

