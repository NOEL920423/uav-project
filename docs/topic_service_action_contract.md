# Topic, service, action, and ownership contract

## Scope and conventions

This contract began as a Phase 1 target. Phase 3 now implements only the three
scene subscriptions and six non-control planning publications identified in
the planning section. Every other endpoint remains a future contract.

QoS notation used in the tables:

- `R/TL/K1`: reliable, transient-local, keep-last depth 1.
- `R/V/K5`: reliable, volatile, keep-last depth 5.
- `BE/V/K5`: best-effort, volatile, keep-last depth 5 (sensor-data style).
- Services/actions use the default reliable service/action profiles; action
  feedback is volatile.

All stamped interfaces use the ROS clock contract in
`coordinate_frames.md`. In simulation this means `use_sim_time=true` and the
same authoritative `/clock`. No spatial message may have an empty frame.

## Scene contracts

| Exact name and type | Owner | Subscribers / clients | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/scene/obstacles` — `uav_interfaces/msg/ObstacleArray` | `uav_scene_bridge` scene publisher | `uav_navigation` planner/validator, visualization, recorder | `header.frame_id=isaac_world`; stamp is the committed scene snapshot time from ROS clock | `R/TL/K1`; once per successful generation/update | Publish only a complete, validated snapshot while episode state is `READY`; all dimensions finite and nonnegative | Do not replace the last valid snapshot with partial/invalid data; generation response/state reports failure and control stays HOLD | Scene adapter phase |
| `/uav/scene/start` — `geometry_msgs/msg/PoseStamped` | `uav_scene_bridge` | planner, recorder, visualization | `isaac_world`; same scene snapshot stamp as obstacles/goal | `R/TL/K1`; once per successful generation/update | Same episode and snapshot as obstacles and goal | Suppress inconsistent snapshot; fail generation and retain HOLD | Scene adapter phase |
| `/uav/scene/goal` — `geometry_msgs/msg/PoseStamped` | `uav_scene_bridge` | planner, tracker, recorder, visualization | `isaac_world`; same scene snapshot stamp | `R/TL/K1`; once per successful generation/update | Same episode and snapshot as obstacles/start | Suppress inconsistent snapshot; planner must not run without a matching goal | Scene adapter phase |
| `/uav/scene/episode_id` — `std_msgs/msg/String` | `uav_scene_bridge` | all episode participants and recorder | No frame/header; exact ID also appears in `EpisodeState` and service/action data. Correlate to the stamped scene snapshot, not DDS receive time | `R/TL/K1`; once per accepted episode, optional on-change repeat | Nonempty and unique for an accepted episode | Empty/duplicate ID rejects generation; no new scene becomes active | Scene adapter phase |
| `/uav/scene/generate` — `uav_interfaces/srv/GenerateEpisode` | Server: `uav_scene_bridge` | Client: episode coordinator / `uav_bringup` composition | No frame/header; response ID binds subsequent stamped `isaac_world` snapshot | Default service QoS; request-driven, at most one active call | Allowed only from `IDLE` or reset-complete state; seed/count validated; `enable_recording` is a request, not proof recording started | Return `success=false` and detail; publish no partial scene and remain/return HOLD | Scene adapter phase |

## Planning contracts

Phase 0 explicitly requires the first milestone's planner/PX4 paths to use
`px4_ned`. The reserved `map` alternative is blocked by the decision in
`coordinate_frames.md`.

Phase 3 implements `path_raw`, `path_simplified`,
`path_bspline_candidate`, `bspline_valid`, `path`, and `status` with `R/TL/K1`
QoS. It requires matching stamped `isaac_world` inputs and replans only for
changed normalized content. A failure publishes empty `px4_ned` paths and
`bspline_valid=false` to clear durable stale data, never a nonempty unsafe final
path. B-spline is candidate generation only; all control endpoints remain
inactive.

| Exact name and type | Owner | Subscribers | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/planner/path_raw` — `nav_msgs/msg/Path` | `uav_navigation` A* adapter | simplifier, validator, debug visualization, recorder | Path and every pose: `px4_ned`; ROS stamp of planning result, with pose stamps equal or explicitly ordered in same clock | `R/TL/K1`; event-driven per planning attempt | Diagnostic raw path only after scene/frame inputs are coherent; not executable by PX4 | Empty/no-path is reported on status; no final path and HOLD remains selected | A* extraction phase |
| `/uav/planner/path_simplified` — `nav_msgs/msg/Path` | `uav_navigation` simplifier | B-spline candidate generator, final selector, validator, recorder | `px4_ned`; stamp inherited from/current planning transaction | `R/TL/K1`; event-driven per successful simplification | Must preserve endpoints and pass continuous segment validation before final selection | On failure, validator may fall back to validated raw path; otherwise no final path and HOLD | Planner decomposition phase |
| `/uav/planner/path_bspline_candidate` — `nav_msgs/msg/Path` | `uav_navigation` B-spline module | continuous validator, debug visualization, recorder | `px4_ned`; candidate-generation ROS stamp | `R/TL/K1`; event-driven, only when smoothing requested | Diagnostic candidate only; never executable merely because it was published | Nonfinite, out-of-bounds, curvature, endpoint, or continuous-clearance failure sets `bspline_valid=false`; final path falls back to validated A* | B-spline phase |
| `/uav/planner/path` — `nav_msgs/msg/Path` | `uav_navigation` final path selector/validator | A* follower candidate generator, recorder, visualization | `px4_ned`; stamp of final validation/selection | `R/TL/K1`; event-driven plus optional <=1 Hz latched republish | Publish only a completely validated raw/simplified/B-spline selection in planning-ready episode state | Never publish unsafe/partial replacement; status fails and command selection remains HOLD | Planner integration phase |
| `/uav/planner/bspline_valid` — `std_msgs/msg/Bool` | B-spline continuous validator | final selector, recorder, visualization | No frame/header; applies to candidate with the current planning transaction/status | `R/TL/K1`; once with each candidate/validation result | `true` only after all continuous safety gates pass | Missing/ambiguous/stale value is false; selector uses validated A* fallback | B-spline phase |
| `/uav/planner/status` — `std_msgs/msg/String` | planning coordinator | episode coordinator, recorder, operator UI | No frame/header; text includes transaction/episode ID; authoritative timing remains stamped paths/state | `R/TL/K1`; on transition/error, optional 1 Hz while planning | Controlled vocabulary plus human detail; not a command channel | Unknown/error status blocks final path acceptance and forces/retains HOLD | Planner decomposition phase |

## Candidate and selected control contracts

For every `TwistStamped` command, `header.frame_id=px4_ned`; linear velocity is
NED and angular Z is the documented NED yaw-rate convention. The header stamp
is mandatory. Expected nominal command rate is 20 Hz. A future mux owns a
configurable short freshness threshold; until validated, the conservative
starting proposal is 0.25 s and is not implemented in Phase 1.

| Exact name and type | Owner | Subscriber | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/control/astar_command` — `geometry_msgs/msg/TwistStamped` | A* path follower only | command multiplexer, recorder | `px4_ned`; ROS clock at command computation | `R/V/K5`; 20 Hz while selected-capable | Only after a valid final path and active episode; bounded finite values | Stale/nonfinite/out-of-bounds command is rejected; mux selects HOLD | Follower phase |
| `/uav/control/joystick_command` — `geometry_msgs/msg/TwistStamped` | joystick node only | command multiplexer, recorder | `px4_ned`; ROS clock at input processing | `R/V/K5`; 20 Hz while teleop enabled | Requires fresh joystick input and deadman; cannot arm/PX4-publish directly | Timeout/deadman release publishes or selects HOLD, then candidate becomes stale | Teleop migration phase |
| `/uav/control/navrl_command` — `geometry_msgs/msg/TwistStamped` | bounded NavRL inference node only | command multiplexer, recorder | `px4_ned`; ROS clock at inference completion | `R/V/K5`; target 20 Hz, bounded by validated inference latency | Deployment/inference only; model loaded/validated, fresh observation, active episode | Late/invalid inference is discarded; never reuse stale output; mux selects HOLD | NavRL deployment phase |
| `/uav/control/hold_command` — `geometry_msgs/msg/TwistStamped` | safety controller only | command multiplexer, recorder | `px4_ned`; ROS clock each safety cycle | `R/V/K5`; 20 Hz whenever graph is control-capable | Always available before any non-HOLD source; bounded zero/position-hold semantics finalized with PX4 adapter | Loss of safety heartbeat blocks PX4 output rather than falling through to another candidate | Safety-controller phase |
| `/uav/control/selected_command` — `geometry_msgs/msg/TwistStamped` | command multiplexer only | PX4 output node, recorder | `px4_ned`; retains selected candidate stamp or a new documented mux stamp; age is measurable | `R/V/K5`; 20 Hz during control-ready states | Exactly one selected source; invalid source selection, stale candidate, or loss of selected candidate results in HOLD | PX4 adapter rejects stale/invalid selected command and enters its safe HOLD/stop-output state | Multiplexer phase |
| `/uav/control/source` — `std_msgs/msg/String` | command multiplexer only | PX4 output node, episode state owner, recorder, UI | No frame/header; source change is paired with next stamped selected command and copied into `EpisodeState.control_source` | `R/TL/K1`; on change plus optional 1 Hz heartbeat | Allowed values initially `HOLD`, `ASTAR`, `JOYSTICK`, `NAVRL`; default `HOLD` | Missing/unknown value is invalid and selects HOLD | Multiplexer phase |

Ownership is exclusive: the A* follower, joystick node, NavRL policy, safety
controller, and mux publish only their named candidate/selected topic. The
future PX4 output node in `uav_px4_control` is the only node allowed to publish
actual `/fmu/in/*` command topics. Planner, joystick, recorder, policy, scene,
camera, and mux nodes may not do so. The mux is not implemented in Phase 1.

## Episode contracts

| Exact name and type | Owner | Subscribers / clients | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/episode/state` — `uav_interfaces/msg/EpisodeState` | episode coordinator | all packages, recorder, UI | Nonspatial header uses empty `frame_id` by contract; ROS stamp is state-transition/heartbeat time | `R/TL/K1`; on transition and 2 Hz heartbeat while active | One monotonic lifecycle per `episode_id`; phase and control source use controlled values; terminal success/collision immutable | Illegal transition produces failure/abort detail and HOLD; no success claim from incomplete cleanup | Episode orchestration phase |
| `/uav/episode/run` — `uav_interfaces/action/RunEpisode` | Action server: episode coordinator | Clients: operator/test harness | No frame/header in action fields; action acceptance time and feedback-associated `EpisodeState` provide ROS-time context | Default action QoS; one active goal, feedback target 2 Hz and on transitions | Validate episode ID/controller source/options before acceptance; cancellation always supported; `enable_bspline` requests validation, not bypass | Reject invalid goal; cancel/abort selects HOLD, stops recording safely, cleans up, and returns `success=false` with detail | Episode orchestration phase |

## Camera contracts

| Exact name and type | Owner | Subscribers | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/fpv/image_raw` — `sensor_msgs/msg/Image` | `uav_camera_bridge` FPV publisher | NavRL inference, recorder, visualization | `uav_fpv_camera`; capture simulation-time stamp | `BE/V/K5`; nominal 10 Hz, configurable/declared | Publish only complete supported encodings with dimensions/step consistent; same stamp as FPV CameraInfo | Drop corrupt/incomplete frame, increment diagnostics; never fabricate or block control thread | Camera bridge phase |
| `/uav/fpv/camera_info` — `sensor_msgs/msg/CameraInfo` | `uav_camera_bridge` FPV calibration owner | same FPV consumers | `uav_fpv_camera`; exact matching image stamp | `BE/V/K5`; with every image at nominal 10 Hz | Calibration dimensions/model must match image and active camera configuration | Drop unmatched pair; policy/recorder rejects image without matching calibration | Camera bridge phase |
| `/uav/observer/image_raw` — `sensor_msgs/msg/Image` | `uav_camera_bridge` observer publisher | recorder, visualization, optional policy | `uav_observer_camera`; capture simulation-time stamp | `BE/V/K5`; nominal 10 Hz, configurable/declared | Same completeness/encoding rules; pairing with FPV uses episode ID + ROS stamp/tolerance | Drop corrupt frame and report diagnostic; no half-pair dataset row | Camera bridge phase |
| `/uav/observer/camera_info` — `sensor_msgs/msg/CameraInfo` | `uav_camera_bridge` observer calibration owner | same observer consumers | `uav_observer_camera`; exact matching image stamp | `BE/V/K5`; with every image at nominal 10 Hz | Calibration/configuration must match active observer attachment mode | Drop unmatched pair and report diagnostic | Camera bridge phase |

## Vehicle-state contracts

| Exact name and type | Owner | Subscribers | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/vehicle/pose` — `geometry_msgs/msg/PoseStamped` | PX4 telemetry adapter in `uav_px4_control` | planner/tracker, safety, recorder, UI | `px4_ned`; source sample converted into common ROS clock | `BE/V/K5`; expected 20–50 Hz, configured to telemetry rate | Publish only finite, valid local pose after origin/frame readiness | Stale/invalid pose marks vehicle state unhealthy, rejects non-HOLD control, and blocks output activation | PX4 telemetry phase |
| `/uav/vehicle/twist` — `geometry_msgs/msg/TwistStamped` | PX4 telemetry adapter | tracker, safety, recorder | `px4_ned`; same source-time conversion policy | `BE/V/K5`; expected 20–50 Hz | Finite velocity with declared NED/angular convention; episode/frame ready | Stale/invalid twist blocks non-HOLD activation and is surfaced in episode status | PX4 telemetry phase |
| `/uav/vehicle/odometry` — `nav_msgs/msg/Odometry` | PX4 telemetry adapter | planner/tracker, safety, recorder, UI | `header.frame_id=px4_ned`, `child_frame_id=base_link`; common ROS stamp | `BE/V/K5`; expected 20–50 Hz | Pose/twist/covariance internally consistent; transform contract validated | Stale/inconsistent odometry marks unhealthy; no extrapolated sample may silently command flight | PX4 telemetry phase |

## Global failure and lifecycle rules

- Every candidate and selected command is stamped. Stale candidates are
  rejected; invalid source selection or loss of the selected command results
  in HOLD.
- Durable scene/path/state data is never interpreted as fresh control input.
- A component must not publish a success terminal state until its required
  cleanup and recording finalization complete.
- A QoS incompatibility, frame mismatch, missing `/clock`, time jump, or
  episode-ID mismatch is a visible health failure, not permission to use the
  last sample indefinitely.
- All rates are target contracts and must become parameters with diagnostic
  measurement when their implementations are introduced.
