# Future BC to PPO plan (no PPO in Phase 1)

The future environment should expose a Gymnasium-like direct simulator API:

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

Observation and action must remain identical or explicitly version-compatible
with BC v0. The environment owns frame conversion, observation validation,
action clipping, collision detection, time limits, scene reset, and simulator
stepping. ROS 2 stays outside the high-frequency loop.

Candidate reward terms, to be measured and ablated later, are progress toward
goal, terminal goal reward, collision penalty, smoothness/control penalty, and
optional clearance penalty. None is implemented or tuned here.

The intended initialization path is:

```text
BC checkpoint
 -> load depth encoder + state encoder + action head into stochastic actor
 -> initialize policy mean near BC output
 -> add a new critic
 -> PPO fine-tuning in parallel simulator environments
```

The actor's final distribution and normalized action mapping will need a new
checkpoint contract version. NavRL's vectorized Isaac/TorchRL pattern,
curriculum concepts, evaluation separation, and PPO batching are useful
references. Its motor-level action, LiDAR/dynamic-obstacle observation, exact
reward, Beta actor, and legacy dependencies must not be inherited without a
separate design and validation phase.

