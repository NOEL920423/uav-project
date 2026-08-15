# A* implementation comparison

## Sources inspected

| Source | Role and evidence | Canonical selection |
|---|---|---|
| `legacy/isaac_direct_pipeline/4.px4_astar.py` | 4,124-line direct Isaac/PX4 runner used as the Phase 0 working source. Contains live-stage geometry, A*, simplification, validation, 2.5D filtering, visualization, logging, and flight control. | Canonical function source for pure A*, safety envelopes, simplification, validation, retry, and metrics-compatible geometry. Flight/runtime portions are excluded. |
| `legacy/isaac_direct_pipeline/5.astar_waypoint_exporter.py` | 4,258-line duplicate with JSON waypoint export. Coordinate, core planner, and overflight function blocks are byte-identical to the direct runner. | Cross-check only; exporter and control code are not migrated. |
| `legacy/isaac_ros2_episode_pipeline/5.astar_ros2_path_publisher.py` | 4,533-line Isaac-side duplicate. Same core planner blocks, plus newer metadata-radius reading and an in-process transient-local `/uav/planned_path` publisher. | Canonical adapter reference for metadata radius preference, ROS time stamping, `px4_ned`, and identity pose orientation. Isaac/builtins publisher is replaced. |
| `legacy/isaac_ros2_episode_pipeline/2.scene_episode_generator.py` | Current high-rise scene generator with stored obstacle metadata. Radius is the yaw-invariant footprint circumradius `0.5*hypot(width, depth)`. | Canonical obstacle geometry producer. Phase 2 consumes its typed ROS representation rather than importing Isaac. |
| `/home/noel_614420090/uav_ros2_ws/src/uav_px4_control` | Contains path followers and a demo `Path` publisher, but no A* search, simplifier, or continuous path validator. | Not a planner source; read-only control/migration reference. |

The three large planner files have distinct whole-file hashes because their
wrappers and runtime responsibilities differ. The selected coordinate blocks,
core A*/simplification/validation blocks, and overflight blocks compare equal.

## Behavior comparison

| Property | Direct runner | Waypoint exporter | ROS/Isaac publisher | Phase 2 decision |
|---|---|---|---|---|
| Obstacle representation | USD-derived bounding circle in `ObstacleInfo` | Same | Same, with metadata-radius preference and live AABB floor | Pure `CircularObstacle`; positive finite radius comes from `ObstacleArray` |
| Coordinate frame | Converts `isaac_world` to `px4_ned` | Same | Same; publishes `px4_ned` | Central pure conversion module; node boundary only |
| Grid | Dense logical rectangular grid, 0.05 m, 2.0 m margin | Same | Same | Preserve defaults; sparse occupied dictionary/set |
| Heuristic | Euclidean grid distance | Same | Same | Preserve |
| Neighbors | 8-connected, cardinal cost 1, diagonal `sqrt(2)` | Same | Same | Preserve |
| Diagonal handling | No explicit occupied-corner gate | Same | Same | Preserve search behavior; require independent continuous validation |
| Planning inflation | `radius + 0.18 + 0.13` | Same | Same | Preserve as explicit functions |
| Grid discretization reserve | Computed for diagnostics but not occupancy | Same | Same | Do not add to physical radius |
| Direct-path bias | Optional, weight 0.07 | Same | Same | Preserve and parameterize |
| Clearance-aware cost | Optional; 0.40 m soft radius, 0.25 weight | Same | Same | Preserve and parameterize |
| Endpoint handling | Reject exact forbidden point; otherwise nearest free rounded cell within 1.0 m | Same | Same | Preserve with structured diagnostics |
| Simplification | RDP 0.05 m, densify max 1.30 m | Same | Same | Preserve stages |
| Fallback | greedy safe, then dense raw | Same | Same | Preserve and revalidate final |
| Continuous validation | Circle-to-segment clearance at planning radius + 0.07 m | Same | Same | Independent validator; every segment plus endpoint/finite/spacing checks |
| Retry | Rebuild occupancy with extra 0.07 m when first result is unsafe/missing | Same | Same | Preserve |
| Short obstacle | Ignore if `top + 0.35 <= 2.0 m` | Same | Same | Preserve exact inclusive threshold and parameterize |
| Output | internal waypoints and MAVLink control | JSON export plus inherited control code | one final `Path`, identity orientation, 1 Hz | Publish raw/simplified/final on contract topics; event-driven |
| Execution evidence | Phase 0 identified as working direct pipeline | Export artifact only | Existing semi-ROS pipeline path source | Regression contract locks properties; no flight claim |

## Function-level source map

- `coordinate_frames.py`: legacy `isaac_to_ned_position` and inverse, extended
  only with verified vector/basis/offset behavior.
- `geometry.py`: legacy `clamp`, `distance_xy`,
  `point_to_segment_distance_2d`, interpolation, and polyline lengths.
- `path_validator.py`: legacy planning/validation radii,
  `segment_clearance_to_obstacle`, `is_segment_safe`, path validation, and 2.5D
  filter, with explicit invalid-input corrections.
- `astar_planner.py`: legacy bounds/grid/index conversion, endpoint recovery,
  deterministic 8-neighbor A*, soft costs, exact endpoint restoration, and
  segment-clearance retry.
- `path_simplifier.py`: legacy RDP, densification, greedy safe fallback, and raw
  fallback.
- `path_metrics.py`: legacy polyline/turn geometry extended to the Phase 2
  geometric metric contract. These metrics do not imply flight smoothness.

No file is copied wholesale because every legacy source mixes planning with
Isaac, logging, visualization, `builtins`, MAVLink, or ROS runtime behavior.
