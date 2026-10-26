# Package layout

- `flight/`: expert/BC lifecycle, policy inference, episode monitoring, and vehicle command ownership.
- `px4/`: message adaptation, output gate, streamer, and odometry.
- `control/`: source arbitration and selected-command validation.
- `diagnostics/`: production timing, readiness/reset helpers, expert monitor, and pure safety fixtures.

Only the expert and BC flight launch files are installed. Standalone offline ROS
harnesses and the physical joystick entry point have been removed.
