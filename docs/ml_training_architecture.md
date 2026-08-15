# UAV ML training architecture

## Scope and classification

ML Phase 1 is a ROS-independent behavior-cloning software pipeline. It does
not contain PPO, PX4 output, OFFBOARD, arm, takeoff, landing, or real-flight
behavior. The first runnable data is explicitly classified as
`synthetic_analytic_depth_software_validation_only`; it is not evidence of UAV
navigation performance.

## Existing component audit

The stable engineering baseline already provides reusable pure Python in
`uav_navigation`:

- `astar_planner.plan_path`: deterministic 8-connected A*, continuous path
  validation, safe simplification, optional B-spline candidate, and A* fallback.
- `trajectory_parameterizer.parameterize_trajectory`: NED path to a bounded,
  timed trajectory with velocity, acceleration, jerk, yaw, and yaw rate.
- `trajectory_sampler.sample_trajectory` and
  `trajectory_tracker.compute_tracking_command`: time sampling and the exact
  bounded follower command used as `ASTAR_EXPERT`.
- `offline_kinematic_plant`: a deterministic non-flight plant useful only for
  fixtures.
- `coordinate_frames`: verified Isaac `(x,y,z-up)` to local NED
  `(north=y,east=x,down=-z)` conversion. Quaternion conversion remains
  deliberately unsupported.

ROS adapters publish scene, camera, trajectory, recorder, and control topics,
but are not imported by `uav_ml`. The canonical control sources are `HOLD`,
`ASTAR_EXPERT`, `HUMAN_JOYSTICK`, and `NAVRL_POLICY`; the future learned source
continues to use the existing velocity-command concept.

Legacy files were inspected and are retained under `legacy/`: the direct and
Isaac-side scene generators, dual-camera setup, PNG recorders,
episode manager, pose logger, and A*/PX4 runner. They can generate scenes and
RGB PNG/CSV records, but the recorder currently proposes wall-time
post-synchronization and the runner contains PX4 behavior. Neither is admitted
as a BC v0 expert dataset source.

The local NavRL reproduction was also inspected. It uses Isaac Sim 2023.1.1
(upstream documents 2023.1.0-hotfix.1), an `IsaacEnv`/TorchRL reset-step loop,
GPU-vectorized environments, a 36-by-4 LiDAR tensor, an 8D vehicle/goal state,
five dynamic-obstacle slots, and PPO with CNN/MLP actor and critic. Its useful
ideas are simulator-direct stepping, tensorized parallel environments,
episode statistics, deterministic resets, and separated evaluation. Its motor
action space, LiDAR observation, reward, old Orbit stack, and PPO network are
not copied into BC v0. The separate host install is Isaac Sim 5.1.0-rc.19 with
Isaac Lab 2.3.2, which reinforces the need for an explicit adapter rather than
assuming API compatibility.

## Training

```text
Isaac Sim (future direct Python adapter; synthetic fixture today)
  -> observation_t
  -> BcPolicyV0
  -> normalized 4D action prediction
  -> denormalize and bound
  -> environment step (future rollout only)
```

Training reads episode containers directly with NumPy/PyTorch. ROS 2 is not in
the data loader, forward pass, loss, optimizer, or checkpoint path. PX4 is not
required.

## Dataset generation

```text
scene snapshot at logical step t
  -> pure A* -> safe B-spline or A* fallback
  -> timed trajectory -> pure bounded follower
  -> expert_action_t
  + depth/velocity/goal_direction captured from the same snapshot t
  -> one synchronized episode sample
  -> advance simulator to t+1
```

The committed generator is an analytic-depth software fixture. A real Isaac
adapter must use the same ordering and write the same versioned dataset
contract. It must not merge independently timestamped PNG and command logs.

## Deployment boundary

```text
trained checkpoint
  -> future ROS 2 inference node
  -> /uav/control/policy_command
  -> existing control mux
```

Only this interface is designed here. No ROS node or publisher is implemented,
and no `/fmu/in/*` topic is introduced. The checkpoint inference wrapper is
limited to preprocessing and returning a bounded NumPy action.

## Real Isaac bridge still required

Real demonstrations are blocked on one integration unit: a simulator-direct
adapter that, within one Isaac simulation callback, obtains a 64x64 metric
depth frame and vehicle state, converts the state to the documented frames,
calls the pure expert for that same step, saves the sample, applies the command
to a simulation-only velocity controller, and reports reset/collision/timeout.
The adapter must target the installed Isaac 5.1/Isaac Lab 2.3 APIs or pin a
different supported runtime. Until that is implemented and validated, the
repository makes no real-dataset or closed-loop-navigation claim.
