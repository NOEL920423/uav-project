# uav_scene_bridge

Future owner of Isaac Sim scene lifecycle and typed scene-state publication.

- Inputs: simulator scene state and episode requests (future).
- Outputs: `/uav/scene/*` topics and scene lifecycle service results (future).
- Must not: plan paths, command PX4, record datasets, or own camera transport.
- Phase 1 behavior: an idle node with simulator access disabled by default.
- Future phase: adapt existing scene helpers behind ROS 2 contracts.
