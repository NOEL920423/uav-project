# uav_interfaces

Typed messages and services used by expert collection and BC flight evaluation:
scene/episode snapshots, timed trajectories, tracking/mux status, PX4 candidate/gate/
stream/flight status, scene generation, explicit enable requests, source selection,
and sole-owner vehicle-command requests.

Runtime behavior lives in the owning packages. The unused `RunEpisode` action and
its `action_msgs` build dependency have been removed. Existing message and service
schemas are unchanged, including `Px4SyntheticTelemetry`, which is also used by the
production live telemetry adapter.
