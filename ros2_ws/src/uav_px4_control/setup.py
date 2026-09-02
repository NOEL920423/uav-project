"""Package the UAV PX4 control scaffold."""

from setuptools import find_packages, setup

package_name = "uav_px4_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "config/control_mux.yaml",
                "config/bc_flight.yaml",
                "config/px4_mapping_gate.yaml",
                "config/px4_setpoint_streamer.yaml",
                "config/px4_sitl_flight.yaml",
            ],
        ),
        (
            "share/" + package_name + "/launch",
            [
                "launch/control_mux_offline.launch.py",
                "launch/control_stack_offline.launch.py",
                "launch/px4_mapping_offline.launch.py",
                "launch/px4_gate_offline.launch.py",
                "launch/px4_boundary_offline.launch.py",
                "launch/px4_sitl_stream.launch.py",
                "launch/px4_sitl_flight.launch.py",
                "launch/bc_flight.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Noel",
    maintainer_email="a0916190423@gmail.com",
    description="Safe Phase 1 PX4 control scaffold.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_mux_node = uav_px4_control.control_mux_node:main",
            "bc_policy_node = uav_px4_control.bc_policy_node:main",
            "bc_flight_supervisor_node = "
            "uav_px4_control.bc_flight_supervisor_node:main",
            "bc_episode_monitor_node = "
            "uav_px4_control.bc_episode_monitor_node:main",
            "px4_mapping_gate_node = "
            "uav_px4_control.px4_mapping_gate_node:main",
            "px4_setpoint_streamer_node = "
            "uav_px4_control.px4_setpoint_streamer_node:main",
            "px4_live_telemetry_adapter = "
            "uav_px4_control.px4_live_telemetry_adapter:main",
            "px4_sitl_doctor = "
            "uav_px4_control.px4_sitl_doctor:main",
            "px4_sitl_stream_monitor = "
            "uav_px4_control.px4_sitl_stream_monitor:main",
            "px4_stream_offline_comparison = "
            "uav_px4_control.px4_stream_fixtures:main",
            "px4_odometry_bridge_node = "
            "uav_px4_control.px4_odometry_bridge_node:main",
            "px4_vehicle_command_owner_node = "
            "uav_px4_control.px4_vehicle_command_owner_node:main",
            "px4_sitl_flight_supervisor_node = "
            "uav_px4_control.px4_sitl_flight_supervisor_node:main",
            "px4_sitl_flight_monitor = "
            "uav_px4_control.px4_sitl_flight_monitor:main",
            "px4_generation_probe = "
            "uav_px4_control.px4_generation_probe:main",
            "runtime_smoke_lifecycle_client = "
            "uav_px4_control.runtime_smoke_lifecycle_client:main",
            "px4_boundary_result_monitor = "
            "uav_px4_control.offline_px4_boundary_harness:monitor_main",
            "synthetic_px4_telemetry = "
            "uav_px4_control.offline_px4_boundary_harness:telemetry_main",
            "control_mux_comparison = "
            "uav_px4_control.control_mux_comparison:main",
            "control_mux_result_monitor = "
            "uav_px4_control.offline_control_mux_harness:monitor_main",
            "px4_control_node = uav_px4_control.px4_control_node:main",
            "synthetic_astar_candidate = "
            "uav_px4_control.offline_control_mux_harness:"
            "astar_publisher_main",
            "synthetic_hold_candidate = "
            "uav_px4_control.offline_control_mux_harness:hold_publisher_main",
            "synthetic_joystick_candidate = "
            "uav_px4_control.offline_control_mux_harness:"
            "joystick_publisher_main",
            "synthetic_navrl_candidate = "
            "uav_px4_control.offline_control_mux_harness:"
            "navrl_publisher_main",
        ],
    },
)
