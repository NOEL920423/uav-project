# Coordinate frame and clock contract

## Status

Phase 2 closes the first planner-milestone decision and implements only the
verified position/vector mapping in one pure module. It does not create a TF
tree and does not claim a complete vehicle orientation convention.

## Stable frame names

| Frame | Meaning in the target architecture | Phase 2 rule |
|---|---|---|
| `isaac_world` | Raw Isaac/Pegasus scene coordinates, poses, obstacles, start, and goal | Never call it ENU unless that is proven. Raw scene messages use this frame. |
| `map` | Reserved ROS planning/world frame for a future TF-consistent architecture | No Phase 2 runtime TF or planner message uses it. |
| `px4_ned` | PX4 local North-East-Down position/velocity convention used by the first integration milestone | Canonical Phase 2 planner frame for obstacles, start, goal, and paths. |
| `base_link` | UAV body frame at the selected vehicle reference point | Body-axis convention and exact prim/link origin require validation. |
| `uav_fpv_camera` | Optical frame for the forward camera | Extrinsic transform from `base_link` and optical-axis convention require calibration. |
| `uav_observer_camera` | Optical frame for the observer camera | Whether this camera is world-fixed or body-attached must be explicit per episode/configuration. |

Empty `frame_id` values are invalid for spatial messages. A consumer must
reject a message whose frame differs from its configured contract unless an
explicit, timestamp-valid TF transform is available.

## Provisional position mapping inherited from Phase 0

The first milestone preserves the empirically used position mapping:

```text
Isaac [x, y, z] -> PX4 local NED [north, east, down] = [y, x, -z]
PX4 local NED [north, east, down] -> Isaac [x, y, z] = [east, north, -down]
```

This is a project-local position mapping, not proof of a general ENU-to-NED
pose transform. It says nothing complete about quaternion handedness, yaw zero,
camera optical axes, vehicle-body axes, local-origin offsets, reset behavior,
or whether the world origin moves between episodes.

Phase 2 resolves the first planning decision in favor of `px4_ned`. Raw scene
geometry remains `isaac_world` and is converted exactly once at the ROS planner
node boundary. Translation parameters `ned_offset_x/y/z` apply to positions
only. Velocity and acceleration vectors use the axis mapping without offsets.
`map` may become a higher-level canonical frame only after deployment origin,
orientation, and TF requirements are validated.

Planar heading vectors follow the same XY swap. Under the explicit conventions
that Isaac yaw is counter-clockwise from Isaac `+X`, and NED yaw is clockwise
from north toward east, `yaw_ned = pi/2 - yaw_isaac`, normalized to
`[-pi, pi)`. This is a planar mathematical contract, not a complete body pose.

Quaternion conversion, body-axis origin, camera optical axes, origin reset,
and covariance semantics remain unsupported. Phase 2 `nav_msgs/Path` uses
identity pose orientation and no Phase 2 consumer may interpret it as heading.

**DECISION REQUIRED BEFORE FLIGHT INTEGRATION:** validate quaternion/body/camera
transforms, origin reset, and covariance behavior against live Pegasus/PX4.

## Clock contract

All target ROS nodes use ROS time from `node.get_clock()` for message header
stamps. In simulation, every participating node must set `use_sim_time=true`
and a single authoritative `/clock` publisher must exist. The graph must not
start an episode until `/clock` is present, advancing, and shared by scene,
image, vehicle-state, command, planning, and episode messages.

In a non-simulation test, every node must set `use_sim_time=false`; ROS system
time then becomes the common header basis. Mixed `use_sim_time` settings are a
configuration error. A zero stamp, backward time jump without lifecycle reset,
or stale stamp causes the affected sample/command to be rejected.

Wall time and monotonic time may be retained as recorder diagnostics and
timeout implementation details. They must not be the primary cross-topic
synchronization key. Dataset joins use `episode_id` plus ROS header stamp; an
image and its CameraInfo share the same stamp, and candidate/selected commands
are freshness-checked using their `TwistStamped.header.stamp` in the same ROS
time domain.

`std_msgs/String`, `std_msgs/Bool`, services, and action fields without a
header do not gain an implicit timestamp. Their correlation and timing rules
must use the associated stamped message or action lifecycle, as documented in
the interface contract.

The Phase 2 planner is input/event-driven and performs no stateful timestamp
arithmetic. It therefore needs no `time_utils.py`; pure tests are independent
of ROS time and `/clock`.
