# Package layout

- `planner/` owns A*, B-spline, geometric validation, and coordinate frames.
- `trajectory/` owns path timing, sampling, validation, and yaw profiles.
- `tracking/` owns the follower and tracking command validation.
- `diagnostics/` contains offline plants, fixtures, harnesses, and comparisons.

Console-script and launch names remain stable; import paths follow this layout.
