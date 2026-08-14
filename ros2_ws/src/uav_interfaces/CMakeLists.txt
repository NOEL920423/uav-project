cmake_minimum_required(VERSION 3.8)
project(uav_interfaces)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/EpisodeState.msg"
  "msg/TimedTrajectory.msg"
  "msg/TrajectoryPoint.msg"
  "msg/TrajectoryTrackingStatus.msg"
  "msg/ControlMuxStatus.msg"
  "msg/Px4OutputGateStatus.msg"
  "msg/Px4SetpointCandidate.msg"
  "msg/Px4StreamStatus.msg"
  "msg/Px4FlightStatus.msg"
  "msg/Px4SyntheticTelemetry.msg"
  "srv/GenerateEpisode.srv"
  "srv/SetControlSource.srv"
  "srv/SetPx4OutputEnable.srv"
  "srv/SetPx4StreamEnable.srv"
  "srv/SetPx4FlightEnable.srv"
  "action/RunEpisode.action"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
