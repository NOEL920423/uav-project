# uav_px4_control

Guarded PX4 control shared by automated expert collection and BC flight evaluation.

- `control/`: source registry, candidate validation, and the sole selected-command mux.
- `px4/`: live telemetry, odometry bridge, coherent mux/gate pairing, mapping, and streaming.
- `flight/`: expert and BC lifecycle supervisors, BC inference, episode monitoring, and vehicle command ownership.
- `diagnostics/`: production timing, readiness/reset helpers, expert result monitoring, and pure test fixtures.

Only two launch files are installed: `px4_sitl_flight.launch.py` for expert collection
and `bc_flight.launch.py` for learned-policy evaluation. Use the root `./uav` entry points.
The expert graph uses planner/follower candidates; BC evaluation uses policy candidates.
Both retain mux HOLD, freshness checks, gate latching, SITL identity, and explicit enable requirements.

`px4_setpoint_streamer_node` exclusively publishes `/fmu/in/trajectory_setpoint`
and `/fmu/in/offboard_control_mode`. `px4_vehicle_command_owner_node` exclusively
publishes `/fmu/in/vehicle_command`. Other modules never bypass these owners.

Source identifiers and typed interfaces retain their existing compatibility.
Standalone offline graphs, synthetic ROS publishers, placeholders, and the hardware
joystick entry point have been removed. Pure safety fixtures remain under test.
See `docs/architecture.md` and `docs/control_issue.md` for the current flow and open issues.
