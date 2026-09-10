# Package layout

- `flight/` owns BC and guarded-flight lifecycle logic.
- `px4/` owns PX4 message adaptation, output gating, streaming, and odometry.
- `control/` owns source arbitration and selected-command validation.
- `diagnostics/` contains offline harnesses, fixtures, monitors, and doctors.

Console-script and launch names remain stable; import paths follow this layout.
