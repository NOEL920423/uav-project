# Phase 1 architecture guardrails

These are release gates, not suggestions. Phase 1 implements only packages,
interfaces, contracts, and harmless placeholders.

1. NavRL high-throughput training remains outside ROS 2.
2. NavRL deployment may later use a bounded ROS 2 inference node.
3. Only one final PX4 output node may command the vehicle.
4. A* planning logic must be separated from PX4 control.
5. B-spline smoothing must be separated from A* search.
6. B-spline candidates must never bypass continuous safety validation.
7. Existing direct-`pymavlink` code remains a legacy fallback until ROS 2
   integration passes testing.
8. The legacy ROS 2 workspace remains untouched until migration is validated.
9. Isaac Sim-specific imports must not leak into pure planner modules.
10. Runtime success must never be claimed from scaffold-only tests.

Additional enforcement rules:

- A placeholder may initialize, declare harmless parameters, log, idle, and
  shut down. It must not access Isaac APIs, open cameras/files, plan/smooth,
  arm, take off, or publish application or PX4 command topics.
- Scene, camera, navigation, policy, joystick, recorder, and command-mux
  components may not publish `/fmu/in/*`. The future safety-gated PX4 output
  adapter is the sole owner.
- Interface definitions do not prove a producer, consumer, transform, timing
  path, or simulator integration works.
- The `px4_msgs` version must match the selected PX4 firmware before an output
  adapter is built. The legacy vendor checkout is not copied as a shortcut.
- No A*, B-spline, multiplexer, camera, recorder, PX4, or flight runtime is in
  scope until the relevant frame, clock, lifecycle, and safety gates are
  reviewed.
