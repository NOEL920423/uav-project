"""Package the UAV data recorder scaffold."""

from setuptools import find_packages, setup

package_name = "uav_data_recorder"

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
    description="Safe Phase 1 data recorder scaffold.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "data_recorder_node = uav_data_recorder.data_recorder_node:main",
            "expert_dataset_recorder = "
            "uav_data_recorder.expert_dataset_recorder_node:main",
            "episode_scene_client = "
            "uav_data_recorder.episode_scene_client:main",
            "visual_qa_capture = "
            "uav_data_recorder.visual_qa_capture:main",
        ],
    },
)
