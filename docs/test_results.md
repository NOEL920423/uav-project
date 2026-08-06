# Test results

## Phase 2 A* migration

Date: 2026-08-06 (Asia/Taipei)

Environment:

- ROS 2 Jazzy from `/opt/ros/jazzy`
- system Python 3.12.3
- branch `feature/ros2-astar-bspline-integration`
- no legacy workspace sourced

### Package build and tests

The incremental pre-commit validation completed with:

```text
colcon build --symlink-install --packages-select uav_navigation
Summary: 1 package finished

colcon test --packages-select uav_navigation
27 pytest items passed

colcon test-result --verbose
Summary: 37 tests, 0 errors, 0 failures, 0 skipped
```

Coverage includes coordinate basis/inverse/offset boundaries, geometry,
planning and validation radii, 2.5D overflight thresholds, deterministic A*,
endpoint rejection/recovery, wide and narrow gaps, bounded no-path, exact
endpoints, safe simplification fallback, geometric metrics, flake8, and pep257.

The final clean whole-workspace result is recorded below.

### Offline ROS 2 integration

Command:

```bash
source /opt/ros/jazzy/setup.bash
source /home/noel_614420090/uav-project/ros2_ws/install/setup.bash
ros2 launch uav_navigation astar_planner_offline.launch.py
```

Observed result:

```text
Published fixed offline obstacle/start/goal
SUCCESS|method=rdp|raw_points=81|simplified_points=6|final_points=6
|length_m=4.538611|minimum_physical_clearance_m=0.460000|fallback=none
offline integration passed: raw=81, simplified=6, final=6, frame=px4_ned
```

The harness independently checked the converted exact endpoints `(0, -2, -2)`
and `(0, 2, -2)`, every final segment against the validation envelope, all
three nonempty path messages, and the absence of `/fmu/in/*`. The fixture exit
triggered a clean launch shutdown of the persistent planner.

Representative durable final-path echo:

```text
header.frame_id: px4_ned
first position: {x: 0.0, y: -2.0, z: -2.0}
last position:  {x: 0.0, y:  2.0, z: -2.0}
pose count: 6
identity orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
```

Topic types observed while the planner remained alive:

```text
/uav/scene/obstacles [uav_interfaces/msg/ObstacleArray]
/uav/scene/start [geometry_msgs/msg/PoseStamped]
/uav/scene/goal [geometry_msgs/msg/PoseStamped]
/uav/planner/path_raw [nav_msgs/msg/Path]
/uav/planner/path_simplified [nav_msgs/msg/Path]
/uav/planner/path [nav_msgs/msg/Path]
/uav/planner/status [std_msgs/msg/String]
```

`ros2 topic list | grep '^/fmu/in/'` returned no lines. No PX4, Isaac Sim,
Pegasus, or XRCE process was launched by the offline graph. A pre-existing
behavior-cloning inference worker (PIDs 153294/153301, started 2026-07-21) uses
an Isaac-distribution Python executable but is unrelated to this launch; its
start time predates this Phase 2 test by 16 days.

### Final clean validation

Existing `build/`, `install/`, and `log/` directories were moved without
deletion to `/tmp/uav-phase2-build.hT2Fi2`. An `env -i` shell used only
`/usr/bin:/bin`, ROS 2 Jazzy, `LANG=C.UTF-8`, and system Python 3.12.3.

```text
colcon build --symlink-install
Summary: 7 packages finished [8.45s]

colcon test
Summary: 7 packages finished [2.27s]

colcon test-result --verbose
Summary: 37 tests, 0 errors, 0 failures, 0 skipped
```

The offline launch was repeated from that isolated clean build and returned the
same successful 81/6/6 path result. Static scans found no ROS, Isaac, PX4, or
MAVLink import in the seven pure modules, no direct PX4 publisher, and no
tracked generated/build/cache/model artifact. The legacy workspace source hash
before and after Phase 2 was identical:

```text
9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb
```

## Phase 3 validated B-spline candidate

Date: 2026-08-06 (Asia/Taipei)

The pure implementation uses system Python 3.12.3 and adds no NumPy, SciPy, or
other runtime dependency. Package and full-workspace results after the Phase 3
changes were:

```text
./uav build --packages-select uav_navigation
Summary: 1 package finished

./uav test --packages-select uav_navigation
Summary: 63 tests, 0 errors, 0 failures, 0 skipped

./uav verify
Summary: 7 packages finished (build)
Summary: 7 packages finished (test)
Summary: 63 tests, 0 errors, 0 failures, 0 skipped
VERIFY SUCCESS: build, tests, interfaces, imports, and safety checks passed
```

Coverage adds clamped basis/De Boor evaluation, effective-degree reduction,
exact endpoints, duplicate removal, arc-length resampling, sample bounds,
continuous clearance and bounds, straight/circular curvature, self-intersection,
planner selection/fallback, determinism, six-scene comparison, and ROS fixture
contracts. Existing Phase 2 regressions remain in the same suite.

Accepted ROS fixture:

```text
./uav offline-check enable_bspline:=true \
  fixture:=bspline-safe-single-obstacle
SUCCESS|astar_success=true|bspline_valid=true|bspline_selected=true
|final_source=BSPLINE|raw_points=81|simplified_points=6
|candidate_points=55|final_points=55|length_m=4.243945
|minimum_physical_clearance_m=0.419106
|maximum_curvature_inverse_m=0.749852
OFFLINE CHECK SUCCESS
```

Rejected ROS fixture, with overall success because the validated A* baseline
remained available:

```text
./uav offline-check enable_bspline:=true \
  fixture:=bspline-rejected-corner-cut
SUCCESS|astar_success=true|bspline_valid=false|bspline_selected=false
|final_source=ASTAR_FALLBACK|raw_points=81|simplified_points=6
|candidate_points=64|final_points=6
|bspline_rejection=segment 19 intersects obstacle offline_tower
OFFLINE CHECK SUCCESS
```

The disabled fixture returned `ASTAR_SIMPLIFIED`. Strict-clearance and
strict-curvature fixtures returned `ASTAR_FALLBACK`. Two-point, three-point,
duplicate-control-point, and self-intersection-preflight fixtures all completed
successfully. Every launch reported `SAFE: no /fmu/in/* topics detected`.

The installed `geometric_path_comparison` entry point completed six fixed
scenes: four candidates accepted as `BSPLINE`, while strict clearance and
curvature gates rejected the candidate and selected `ASTAR_FALLBACK`. Full
metrics and limitations are recorded in `phase3_geometric_comparison.md`.

The canonical legacy hash (pruning nested `.git` directories) remained exactly
the same before and after Phase 3, and that workspace was never sourced or
built:

```text
9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb
```
