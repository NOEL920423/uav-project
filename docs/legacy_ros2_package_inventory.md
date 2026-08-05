# Legacy ROS 2 package inventory

## Scope and preservation evidence

This inventory is a read-only audit of
`/home/noel_614420090/uav_ros2_ws/src`. The legacy workspace was not built,
cleaned, moved, or edited. Its `build/`, `install/`, and `log/` directories were
excluded. Before Phase 1 edits, a deterministic hash over files below `src/`
was recorded as:

```text
9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb  -
```

The same hash must be reproduced at Phase 1 completion.

## Packages found

| Package | Full path | Build type | Dependencies | Disposition |
|---|---|---|---|---|
| `px4_msgs` 2.0.1 | `/home/noel_614420090/uav_ros2_ws/src/px4_msgs` | `ament_cmake` / `rosidl` interface package | `builtin_interfaces`, `ros_environment`, `rosidl_default_generators`, `rosidl_default_runtime` | External/vendor dependency. Retain this checkout read-only; do not copy it into the tracked workspace. Select a PX4-compatible release before PX4 migration. |
| `uav_px4_control` 0.5.0 | `/home/noel_614420090/uav_ros2_ws/src/uav_px4_control` | `ament_python` | `rclpy`, `px4_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `std_msgs`, `std_srvs`, `launch`, `launch_ros` | Mixed legacy implementation. Retain read-only for behavior/regression evidence; selectively migrate safety concepts and replace direct command ownership later. |

`px4_msgs` contains generated-interface sources only. It has no executable,
node, launch file, publisher, subscriber, service, action, or ROS parameter.
Camera-related files found there (`CameraStatus.msg`, `CameraCapture.msg`, and
`CameraTrigger.msg`) are PX4 message definitions, not a camera publisher.

## `uav_px4_control` executables

| Executable / source | Publishers | Subscribers | Services / actions | Parameters or CLI | PX4 and migration decision |
|---|---|---|---|---|---|
| `simple_offboard` / `simple_offboard.py` | `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command` | `/fmu/out/vehicle_odometry` | None / none | No declared ROS parameters | Direct PX4 output. Unsafe as a target owner; retain read-only as a smoke-test reference and deprecate later. |
| `waypoint_follower` / `waypoint_follower.py` | Same three `/fmu/in/*` topics | `/fmu/out/vehicle_local_position` and `/fmu/out/vehicle_odometry` (duplicate subscriptions with alternate QoS) | None / none | `waypoint_file` | Direct PX4 output. Replace with planner contract + single output owner; retain read-only until regression coverage exists. |
| `lookahead_follower` / `lookahead_follower.py` | Same three `/fmu/in/*`; `/uav_control/mission_state` | `/uav/planned_path`, `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_odometry`, `/fmu/out/vehicle_status`, `/fmu/out/vehicle_command_ack`, `/fmu/out/vehicle_land_detected` | `/uav_control/start_mission`, `reset_mission`, `abort_mission`, `get_status`; no actions | `path_topic`, `expected_frame_id`, `allow_empty_frame_id`, `auto_start_on_path`, `final_goal_radius_xy_m`, `final_goal_radius_z_m`, `final_goal_hold_s`, `position_timeout_s`, `vehicle_status_timeout_s`, `prestream_s`, `activation_timeout_s`, `landing_timeout_s`, `command_retry_s`, `max_yaw_rate_radps`, `yaw_alignment_min_speed_mps`, `path_lookahead_m` | Direct PX4 output. Migrate its bounded lifecycle, ACK, timeout, and landing concepts, but replace publication ownership. |
| `mission_orchestrator` / `mission_orchestrator.py` | `/uav_mission/state` | `/uav_control/mission_state` | Servers: `/uav_mission/run_episode`, `abort`, `get_status`. Clients: `/uav_control/reset_mission`, `start_mission`, `abort_mission`, `/uav_sim/prepare_episode`, `start_recording`, `stop_recording`, `start_pose_logger`, `stop_pose_logger`. No actions. | `ready_timeout_s`, `mission_timeout_s`, `abort_timeout_s`, `service_timeout_s`, `follower_transition_timeout_s` | Migrate/replace with typed episode lifecycle and `RunEpisode` action. Its `/uav_sim/*` servers are not in legacy `src/`. |
| `joystick_teleop` / `joystick_teleop.py` | Same three `/fmu/in/*` topics | `/joy`, `/fmu/out/vehicle_local_position` | None / none | Fixed constants: 20 Hz, 0.5 s timeout, speed/yaw bounds, axes/buttons, deadman and arm/land/disarm bindings; no ROS parameters | Direct PX4 output. Later migrate input semantics so it publishes only `/uav/control/joystick_command`; deprecate direct ownership. |
| `udp_joy_bridge` / `udp_joy_bridge.py` | `/joy` | UDP JSON packets, not a ROS topic | None / none | CLI: `--listen-host`, `--port` (5005), `--topic`, `--rate` (30 Hz), `--timeout`, axes/buttons | Retain as an input bridge candidate; it must never own PX4 output. |
| `demo_path_publisher` / `demo_path_publisher.py` | `/uav/planned_path` (`nav_msgs/Path`) | None | None / none | `path_topic`, `frame_id=px4_ned`, `publish_rate_hz=1.0` | Retain as a read-only test reference; replace topic with the new planning contract later. |
| `bc_training_manager` / `bc_training_manager.py` | `/uav_bc/training_status` | None | `/uav_bc/train`, `/uav_bc/get_status`, `/uav_bc/cancel`; no actions | `training_python`, `python_module_root`, `dataset_root`, `output_dir`, `episode_glob`, `epochs`, `batch_size`, `horizon_s`, `validation_fraction` | Deprecate the ROS high-throughput training manager. Training remains outside ROS 2. Keep read-only for experiment provenance. |
| `bc_flight_controller` / `bc_flight_controller.py` | Inherits all direct PX4 and mission-state publishers from `lookahead_follower` | Inherits follower telemetry/path inputs; adds `/uav_sim/episode_id` | Inherits the four `/uav_control/*` services; no actions | Inherited follower parameters plus `prediction_file`, `bc_max_speed_xy_mps`, `bc_max_speed_z_mps`, `bc_position_horizon_s`, `bc_image_timeout_s`, `bc_mission_timeout_s`, `bc_progress_timeout_s`, `bc_takeoff_altitude_m`, `bc_path_lookahead_m`, `bc_path_velocity_blend`, `bc_cross_track_kp`, `bc_cross_track_max_correction_mps`, `bc_altitude_kp` | Retain as read-only research evidence. Replace with a bounded NavRL/policy candidate publisher; it may not command PX4 directly. |

The non-node modules `bc_dataset.py`, `bc_inference_worker.py`, `bc_model.py`,
and `bc_train.py` expose no ROS graph interfaces. They remain research
references; training stays outside ROS 2.

## Launch inventory

- `astar_path_mission.launch.py`: starts `lookahead_follower` and
  `mission_orchestrator`; defaults to `/uav/planned_path`, requires
  `px4_ned`, and keeps unsafe `auto_start_on_path` false.
- `bc_path_mission.launch.py`: starts `bc_flight_controller` and
  `mission_orchestrator` with a prediction JSON path; auto-start remains false.
- `bc_training.launch.py`: starts the legacy training manager.
- `joystick_teleop.launch.py`: starts UDP joystick bridge and direct-PX4
  joystick teleop.
- `path_topic_test.launch.py`: starts `demo_path_publisher`.
- `px4_smoke_test.launch.py`: starts direct-PX4 `simple_offboard`.
- `waypoint_mission.launch.py`: starts the JSON waypoint follower and defaults
  to a file outside `src/`.

Backup files and `__pycache__` entries exist in the legacy package. Neither is
authoritative and neither was copied.

## Requested implementation search

| Requested concept | Result inside legacy `src/` |
|---|---|
| `astar_path_mission.launch.py` | Exact launch file found and inspected. |
| `ros2_joystick_teleop_px4.py` | Exact name absent; equivalent behavior is `joystick_teleop.py`. |
| `ros2_lookahead_follower_checked.py` | Exact name absent; equivalent checked follower is `lookahead_follower.py`. |
| `ros2_uav_pose_publisher_logger.py` | Absent. Phase 0 found the implementation in project scripts, not this workspace. |
| episode manager | `mission_orchestrator.py` is a client/orchestrator; no Isaac episode-manager server is present. |
| `/uav_sim/prepare_episode` | Client use found in `mission_orchestrator.py`; server absent. |
| `/uav_sim/stop_all` | No occurrence. |
| camera publishing | No ROS image/CameraInfo publisher; only PX4 camera message definitions. |
| rosbag recording | No rosbag node, service, or action implementation. |
| PX4 offboard control | Found in `simple_offboard`, `waypoint_follower`, `lookahead_follower`, `joystick_teleop`, and inherited by `bc_flight_controller`. |

## Direct `/fmu/in/*` risk summary

Five executable paths can publish flight commands directly:
`simple_offboard`, `waypoint_follower`, `lookahead_follower`,
`joystick_teleop`, and `bc_flight_controller` by inheritance. This violates
the target single-owner rule. None is copied into the new workspace. A future
PX4 migration must first select a compatible `px4_msgs` release and introduce
one safety-gated output node; no other target component may publish these
topics.
