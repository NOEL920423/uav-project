"""Package the guarded expert and BC flight control nodes."""

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
                "launch/px4_sitl_flight.launch.py",
                "launch/bc_flight.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Noel",
    maintainer_email="a0916190423@gmail.com",
    description=(
        "Guarded PX4 flight control for expert collection and BC evaluation."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_mux_node = uav_px4_control.control.control_mux_node:main",
            "bc_policy_node = uav_px4_control.flight.bc_policy_node:main",
            "bc_flight_supervisor_node = "
            "uav_px4_control.flight.bc_flight_supervisor_node:main",
            "bc_episode_monitor_node = "
            "uav_px4_control.flight.bc_episode_monitor_node:main",
            "px4_mapping_gate_node = "
            "uav_px4_control.px4.px4_mapping_gate_node:main",
            "px4_setpoint_streamer_node = "
            "uav_px4_control.px4.px4_setpoint_streamer_node:main",
            "px4_live_telemetry_adapter = "
            "uav_px4_control.px4.px4_live_telemetry_adapter:main",
            "px4_sitl_doctor = "
            "uav_px4_control.diagnostics.px4_sitl_doctor:main",
            "px4_odometry_bridge_node = "
            "uav_px4_control.px4.px4_odometry_bridge_node:main",
            "px4_vehicle_command_owner_node = "
            "uav_px4_control.flight.px4_vehicle_command_owner_node:main",
            "px4_sitl_flight_supervisor_node = "
            "uav_px4_control.flight.px4_sitl_flight_supervisor_node:main",
            "px4_sitl_flight_monitor = "
            "uav_px4_control.diagnostics.px4_sitl_flight_monitor:main",
            "px4_generation_probe = "
            "uav_px4_control.diagnostics.px4_generation_probe:main",
            "runtime_smoke_lifecycle_client = "
            "uav_px4_control.diagnostics.runtime_smoke_lifecycle_client:main",
        ],
    },
)
