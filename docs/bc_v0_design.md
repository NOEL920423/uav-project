# Behavior Cloning v0 design

## Learning problem

```text
observation_t = depth + NED velocity + body-FRD goal direction
BcPolicyV0(observation_t) -> normalized predicted action_t
```

The target is the existing A*/B-spline trajectory follower's bounded
`[v_north,v_east,v_down,yaw_rate]` command. MSE is applied after per-component
action normalization derived from training episodes only. Reporting is in
physical units and includes total MSE plus per-action MSE, MAE, and RMSE.

## Network

`BcPolicyV0` contains:

```text
1x64x64 depth
 -> Conv(1,8,5,stride=2) -> Conv(8,16,3,stride=2)
 -> Conv(16,24,3,stride=2) -> adaptive 4x4 -> Linear(384,32)

velocity[3] + goal_direction[3]
 -> Linear(6,32) -> Linear(32,32)

concat[64] -> Linear(64,64) -> Linear(64,4)
```

Default parameter count: 22,876. It is intentionally small and has no
Transformer. The encoders and action head can later initialize a PPO actor;
the BC checkpoint contains no critic or PPO optimizer.

## Trainer and checkpoint

`python -m uav_ml.train_bc` supports dataset path, epochs, batch size, learning
rate, device (`auto`, CPU, or CUDA), random seed, checkpoint/history paths, and
optional resume. It seeds Python, NumPy, Torch CPU/CUDA, uses a seeded loader,
prints epoch metrics, writes CSV history, and reload-checks the final
checkpoint.

The checkpoint includes model state/config, both contract versions,
normalization statistics, Git SHA, dataset version/statistics, Python/NumPy/
Torch/split seeds, optimizer config/state, device, parameter count, and initial
and final metrics. Raw weights alone are not a valid checkpoint.

## Inference and rollout boundary

`BcPolicyInference` validates shapes/values, applies checkpoint normalization,
runs `eval()` under inference mode, denormalizes, and reapplies action bounds.
It has no ROS or PX4 imports.

A future simulator rollout must remain separate from training and implement an
episode timeout, collision termination, goal termination, invalid observation/
action termination, and the exact action bounds. No closed-loop rollout exists
in Phase 1, so navigation success is not reported.

Future rollout metrics are success rate, collision rate, episode length, final
goal distance, path length, travel time, mean/max action magnitude, and command
smoothness. Open-loop BC metrics are action MSE, MAE, and RMSE.

