# Coordinate frame and clock contract

## Status

This document fixes names and validation rules for Phase 1. It does not create
a TF tree or implement conversions. Phase 0 established a working position
mapping but also found that the project has no canonical transform module and
does not yet define a complete orientation convention.

## Stable frame names

| Frame | Meaning in the target architecture | Phase 1 rule |
|---|---|---|
| `isaac_world` | Raw Isaac/Pegasus scene coordinates, poses, obstacles, start, and goal | Never call it ENU unless that is proven. Raw scene messages use this frame. |
| `map` | Reserved ROS planning/world frame for a future TF-consistent architecture | Name is reserved, but its transform to `isaac_world` and `px4_ned` is unresolved. No Phase 1 runtime publisher uses it. |
| `px4_ned` | PX4 local North-East-Down position/velocity convention used by the first integration milestone | Per Phase 0, planner paths and candidate commands use this frame until the `map` decision is closed. |
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

**DECISION REQUIRED BEFORE PHASE 2:** choose whether `map` becomes the
canonical planning frame or the first migration continues to plan directly in
`px4_ned`. If `map` is selected, specify and test the full
`isaac_world -> map -> px4_ned` transform chain. Until that decision, the
least-destructive contract keeps Phase 0 planner paths in `px4_ned` and raw
scene geometry in `isaac_world`.

**DECISION REQUIRED BEFORE PHASE 2:** define orientation, yaw, body-axis,
camera optical-axis, translation-offset, origin-reset, and covariance
semantics. Tests must cover forward/inverse round trips, known basis vectors,
nonzero offsets, and heading examples before any flight-output adapter is
allowed.

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
