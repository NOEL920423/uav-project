# BC v0 observation and action contract

Contract versions:

- observation: `uav_bc_depth_state_v0.1`
- action: `uav_velocity_yaw_rate_ned_v0.1`

## Observation at step t

| Field | Shape | dtype | Units | Frame | Validation and preprocessing |
|---|---:|---|---|---|---|
| `depth` | `1 x 64 x 64` | `float32` | metres | forward camera optical raster | Positive finite values; clip to `[0.2, 20.0]`; no resize in the loader |
| `velocity` | `3` | `float32` | m/s | `px4_ned`: north, east, down | Finite; clip to `[-5, 5]` before inference |
| `goal_direction` | `3` | `float32` | unitless unit vector | body FRD: forward, right, down | Finite, nonzero, renormalized to unit length |

The depth preprocessing expected from Isaac is: obtain metric range/depth for
the same simulator step, resize once to 64x64 using a documented
depth-preserving method, store channel-first float32 metres, and then clip.
RGB and previous action are intentionally excluded from v0.

Missing depth, non-positive depth, NaN/Inf, wrong shapes, or a zero goal vector
are hard errors. They are never silently replaced. During a future rollout,
the environment must terminate the episode on such an error.

Normalization is `(x - train_mean) / max(train_std, 1e-6)`. Depth uses one
global scalar mean/std; velocity, goal direction, and action use per-component
statistics. Only training episodes contribute. The checkpoint stores all
statistics, and validation/inference reuse them unchanged.

## Coordinate transform for goal direction

Let `d=[north,east,down]` be the normalized NED vector from current vehicle
position to goal, and `psi` the NED yaw. Body FRD is:

```text
forward =  cos(psi) * north + sin(psi) * east
right   = -sin(psi) * north + cos(psi) * east
down    =  down
```

This explicitly avoids treating Isaac world, ENU, NED, and body vectors as
interchangeable.

## Expert and predicted action

```text
[v_north, v_east, v_down, yaw_rate]
```

The first three values are m/s in `px4_ned`; yaw rate is rad/s, positive in the
documented NED yaw convention. The source target is the selected output of
`uav_navigation.trajectory_tracker.compute_tracking_command`, after its
horizontal, vertical, total-speed, acceleration, yaw-rate, and yaw-acceleration
bounds. It is not an A* waypoint.

Inference reapplies conservative static bounds:

- horizontal speed <= 2.0 m/s;
- vertical speed magnitude <= 1.0 m/s;
- total 3D speed <= 2.0 m/s;
- yaw-rate magnitude <= 1.5 rad/s.

BC v0 never predicts motors, PWM, attitude, thrust, or PX4 messages.

