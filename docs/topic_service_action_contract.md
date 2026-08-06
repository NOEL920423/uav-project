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
is mandatory. Phase 6 mux input QoS is reliable/volatile/keep-last 5 and its
output rate is 50 Hz by default. Freshness is based on mux receipt time, never
on a candidate-controlled header stamp; per-source timeouts are configurable.

| Exact name and type | Owner | Subscriber | Frame and timestamp | QoS / rate | State restriction | Failure behavior | Future phase |
|---|---|---|---|---|---|---|---|
| `/uav/control/astar_command` — `geometry_msgs/msg/TwistStamped` | A* path follower only | command multiplexer, recorder | `px4_ned`; ROS clock at command computation | `R/V/K5`; 20 Hz while selected-capable | Only after a valid final path and active episode; bounded finite values | Stale/nonfinite/out-of-bounds command is rejected; mux selects HOLD | Follower phase |
| `/uav/control/joystick_command` — `geometry_msgs/msg/TwistStamped` | joystick node only | command multiplexer, recorder | `px4_ned`; ROS clock at input processing | `R/V/K5`; 20 Hz while teleop enabled | Requires fresh joystick input and deadman; cannot arm/PX4-publish directly | Timeout/deadman release publishes or selects HOLD, then candidate becomes stale | Teleop migration phase |
| `/uav/control/navrl_command` — `geometry_msgs/msg/TwistStamped` | bounded NavRL inference node only | command multiplexer, recorder | `px4_ned`; ROS clock at inference completion | `R/V/K5`; target 20 Hz, bounded by validated inference latency | Deployment/inference only; model loaded/validated, fresh observation, active episode | Late/invalid inference is discarded; never reuse stale output; mux selects HOLD | NavRL deployment phase |
| `/uav/control/hold_command` — `geometry_msgs/msg/TwistStamped` | safety controller only | command multiplexer, recorder | `px4_ned`; ROS clock each safety cycle | `R/V/K5`; 20 Hz whenever graph is control-capable | Always available before any non-HOLD source; bounded zero/position-hold semantics finalized with PX4 adapter | Loss of safety heartbeat blocks PX4 output rather than falling through to another candidate | Safety-controller phase |
| `/uav/control/selected_command` — `geometry_msgs/msg/TwistStamped` | `control_mux` only | Phase 6 offline plant; future PX4 output node and recorder | `px4_ned`; stamp is always the mux ROS clock at publication, never copied from a candidate | `R/V/K5`; configurable, default 50 Hz | Exactly one selected source; each candidate and final selected command are independently validated | Any stale/invalid selected source or validator failure produces internal exact-zero HOLD; no automatic movement-source failover | Phase 6 offline mux |
| `/uav/control/source` — `std_msgs/msg/String` | `control_mux` only | future PX4 output node, episode state owner, recorder, UI | No frame/header; published with every mux cycle | `R/V/K5`; default 50 Hz | Exact values `HOLD`, `ASTAR_EXPERT`, `HUMAN_JOYSTICK`, `NAVRL_POLICY`; startup is `HOLD` | Unknown request is rejected and fail-closes to latched HOLD | Phase 6 offline mux |
| `/uav/control/mux_status` — `uav_interfaces/msg/ControlMuxStatus` | `control_mux` only | test monitor, recorder, operator UI | `header.frame_id=px4_ned`; mux ROS clock | `R/V/K5`; default 50 Hz | Reports request/active source, HOLD reason, handoff time, source health/age, bounds and transition count | Diagnostics never authorize motion; contradictions fail closed | Phase 6 offline mux |
| `/uav/control/set_source` — `uav_interfaces/srv/SetControlSource` | Server: `control_mux` | Client: offline harness; future episode coordinator/UI | No request stamp; service handling uses mux receipt time and current registry health | Default service QoS | Only canonical sources accepted; movement-to-movement changes honor dwell and HOLD barrier; a fresh explicit request is required after a latched fault | Rejected requests return `accepted=false`; invalid requests select `HOLD_INVALID_SOURCE` | Phase 6 offline mux |

Ownership is exclusive: the A* follower, joystick node, NavRL policy, safety
controller, and mux publish only their named candidate/selected topic. The
future PX4 output node in `uav_px4_control` is the only node allowed to publish
actual `/fmu/in/*` command topics. Planner, joystick, recorder, policy, scene,
camera, and mux nodes may not do so. Phase 6 implements only candidate
arbitration and `selected_command`; it contains no PX4 output publisher.

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

## Phase 4 trajectory contracts

Phase 4 is event-driven and uses reliable transient-local depth-one QoS. It is
strictly downstream of the final Phase 3 selection and is not a command path.

| Exact name and type | Owner | Input/source | Frame and timestamp | Acceptance and failure behavior |
|---|---|---|---|---|
| `/uav/trajectory/candidate` — `uav_interfaces/msg/TimedTrajectory` | `uav_navigation` trajectory parameterizer | Only `/uav/planner/path`; pose stamps and orientations ignored | Header and `source_path_frame` are `px4_ned`; output header uses current ROS time | Published only for a finite structured candidate. `valid=true` only after independent validation; rejected finite candidates carry `valid=false` and status. |
| `/uav/trajectory/valid` — `std_msgs/msg/Bool` | Independent trajectory validation result publisher | Current unique path attempt | No header; paired with current candidate/status transaction | Published for every attempted non-identical path. Missing, stale, or rejected is never interpreted as true. |
| `/uav/trajectory/status` — `std_msgs/msg/String` | Trajectory parameterizer | Current unique path attempt | No header; controlled pipe-delimited fields | Reports success/rejection, counts, duration, time scale, dynamics maxima, and explicit rejection reason. |

`TimedTrajectory` positions exactly equal the adjacent-duplicate-cleaned source
path. Time and arc length are finite and strictly increasing. The node consumes
no raw/simplified/candidate planner topics, vehicle state, joystick, NavRL,
simulator, or PX4 topics. No Phase 4 owner publishes `/fmu/in/*`.

## Phase 5 offline tracking contracts

Phase 5 consumes only the accepted Phase 4 candidate/validity pair and offline
`px4_ned` odometry. Its reliable volatile 20 Hz outputs are ROS-level
controller candidates and diagnostics, never PX4 setpoints.

| Exact name and type | Owner | Meaning and frame | Failure behavior |
|---|---|---|---|
| `/uav/control/astar_command` — `geometry_msgs/msg/TwistStamped` | `trajectory_follower_node` | `px4_ned`; linear X/Y/Z are north/east/down velocity, angular Z is NED yaw rate, angular X/Y are zero | Missing, stale, false, wrong-frame, non-finite, time-jump, excessive-error, terminal-timeout, or validator failure selects an exact zero HOLD with a reason |
| `/uav/control/astar_reference_pose` — `geometry_msgs/msg/PoseStamped` | `trajectory_follower_node` | Current interpolated reference position/yaw in `px4_ned` | Absent while no valid reference can be sampled |
| `/uav/control/astar_reference_twist` — `geometry_msgs/msg/TwistStamped` | `trajectory_follower_node` | Current interpolated NED velocity and yaw rate | Absent while no valid reference can be sampled |
| `/uav/control/astar_tracking_status` — `uav_interfaces/msg/TrajectoryTrackingStatus` | `trajectory_follower_node` | Stamped `px4_ned` state, gates, errors, saturation flags, diagnostics, and reason | Explicit waiting/HOLD/terminal state; never silently reuses stale evidence |

The candidate command, selected command, and PX4 output command are three
different ownership layers. Phase 5 implements only the first: a validated A*
follower candidate on `/uav/control/astar_command`. The internal pure
`selected_command` field means only the bounded candidate selected over the
unbounded calculation in the same follower cycle; it is not mux arbitration.
Phase 6 now implements the source mux that owns
`/uav/control/selected_command`; only a separate future `uav_px4_control`
adapter may map that output to `/fmu/in/*`. That PX4 output layer does not
exist in the Phase 6 graph.

Receipt time plus `trajectory_start_delay_s` defines the local tracking epoch;
trajectory timestamps remain relative. Duplicate identical trajectories do
not reset the epoch. Backward/equal control time fails closed and fresh
trajectory, validity, and odometry synchronization is required after a
backward jump.

## Phase 6 offline control mux contracts

The mux subscribes to all four candidate topics but movement ownership stays
exclusive. Startup and explicit `HOLD` requests use an internal exact-zero
command. Requests for a movement source require a fresh, finite, bounded
`px4_ned` candidate. Movement-to-movement changes pass through an exact-zero
HOLD barrier; a target that becomes unhealthy cancels the handoff. Selected
source faults latch HOLD and cannot recover merely because messages resume.

The independent selected-command validator enforces horizontal, vertical,
total speed, acceleration, yaw-rate, yaw-acceleration, frame, finiteness and
monotonic-time gates. Safety HOLD is immediate and exact zero; ordinary rate
limits do not delay it. Unselected-source faults never replace a healthy active
source, and the mux never auto-fails over to another movement source.

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
