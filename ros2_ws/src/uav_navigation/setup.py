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
        ("share/" + package_name + "/config", ["config/astar_planner.yaml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/astar_planner_offline.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Noel",
    maintainer_email="a0916190423@gmail.com",
    description=(
        "Validated A* planner with optional Phase 3 B-spline candidate."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigation_node = uav_navigation.navigation_node:main",
            "astar_planner_node = uav_navigation.astar_planner_node:main",
            "astar_offline_harness = "
            "uav_navigation.astar_planner_node:offline_harness_main",
            "geometric_path_comparison = "
            "uav_navigation.geometric_comparison:main",
        ],
    },
)
