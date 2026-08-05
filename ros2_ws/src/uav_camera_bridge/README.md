# uav_camera_bridge

Future owner of Isaac camera acquisition and ROS image publication.

- Inputs: FPV and observer render products plus calibration (future).
- Outputs: typed `sensor_msgs/Image` and `CameraInfo` topics (future).
- Must not: plan paths, select commands, send PX4 inputs, or write datasets.
- Phase 1 behavior: an idle node with camera access disabled by default.
- Future phase: add explicit image encoding, timestamp, frame, and QoS handling.
