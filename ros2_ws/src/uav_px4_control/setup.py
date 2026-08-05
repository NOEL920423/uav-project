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
            "px4_control_node = uav_px4_control.px4_control_node:main",
        ],
    },
)
