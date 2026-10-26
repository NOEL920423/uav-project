"""Package the UAV navigation planner."""

from setuptools import find_packages, setup

package_name = "uav_navigation"

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
                "config/astar_planner.yaml",
                "config/task.yaml",
                "config/trajectory_follower.yaml",
                "config/trajectory_parameterizer.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "PyYAML>=6,<7"],
    zip_safe=True,
    maintainer="Noel",
    maintainer_email="a0916190423@gmail.com",
    description=(
        "Expert-flight planning, timed trajectories, and tracking."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "astar_planner_node = "
            "uav_navigation.planner.astar_planner_node:main",
            "trajectory_parameterizer_node = "
            "uav_navigation.trajectory.trajectory_parameterizer_node:main",
            "trajectory_follower_node = "
            "uav_navigation.tracking.trajectory_follower_node:main",
        ],
    },
)
