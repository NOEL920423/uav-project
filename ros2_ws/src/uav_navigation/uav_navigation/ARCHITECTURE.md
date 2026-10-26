# Package layout

- `planner/`: expert A*, B-spline, geometry, and coordinate conversion.
- `trajectory/`: timing, sampling, validation, and yaw profiles.
- `tracking/`: expert follower and command validation.
- `diagnostics/`: pure test fixtures for geometry, plant, and tracking metrics.

The expert launch is maintained in `uav_px4_control/launch/px4_sitl_flight.launch.py`.
No standalone offline ROS graph is installed by this package.
