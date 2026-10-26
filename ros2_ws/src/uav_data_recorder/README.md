# uav_data_recorder

`expert_dataset_recorder` records synchronized expert images, odometry, actions,
scene/seed metadata, and safety evidence under `datasets/<dataset>/episode_*`.
It is enabled by the expert flight launch; BC evaluation does not start the recorder.

`episode_scene_client` verifies landed, disarmed, no-failsafe PX4 state before requesting
a seeded Isaac scene. The root `./uav expert-collect` orchestrator owns episode collection,
resume manifests, finalization, validation, and visual QA.

The unused placeholder recorder has been removed. Dataset contracts and recorded
labels are unchanged. This package never commands PX4 or trains models.
See `docs/expert_dataset_collection.md`.
