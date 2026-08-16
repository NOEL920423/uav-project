# Phase 10C: canonical high-rise scene and camera recovery

## Scope and status

Phase 10C ports only the historically validated scene, lighting, and camera
research behavior into the current Phase 10B automation. The legacy episode
pipeline is not reactivated. The current ROS 2 A*, B-spline, follower, mux,
safety gate, PX4 streamer, synchronization, and BC V1 contracts remain owners
of flight and data behavior.

Three visual-QA flights have run successfully. Their ignored artifacts await
the user's image-quality confirmation, so this phase is intentionally not
committed or pushed and no dataset collection has started.

## Canonical historical sources

The following files were read in full before the port:

- `legacy/isaac_ros2_episode_pipeline/2.scene_episode_generator.py`, SHA-256
  `eae160e2284efb5cd0ee474e5e95a0e33d6f6c2ea33ea564f73bec8dbcd10bca`
- `legacy/isaac_ros2_episode_pipeline/1.dual_uav_camera.py`, SHA-256
  `4975fa028e9526893d1e23d32c59fff5b2a5bb1f3747dac3a848d33a67b6f796`
- `legacy/isaac_ros2_episode_pipeline/6.isaac_ros2_episode_manager.py`,
  SHA-256
  `c743c7a3081ea38c44b6c90e9874181a46a99f2eeac668c0f1572aec475b1a3d`

The first camera file supplies the controller behavior and the manager supplies
the effective runtime overrides: FPV forward axis X, look-down -0.8 m, and
Observer mode TOP.

## Recovered scene contract

`isaac/runtime/episode_scene.py` ports the canonical random-call order,
guaranteed-blocker placement, clearances, and rejection sampler. It uses a
local seeded RNG so layouts remain deterministic without changing global RNG
state.

| Parameter | Canonical value |
|---|---:|
| Buildings | 8 |
| Start / goal XY | (0, 0) / (3, 5) m |
| Spawn X / Y | [-2, 5] / [-1, 7] m |
| Width / depth | 0.46--0.72 m |
| Height | 2.80--5.20 m |
| Yaw | -35 to +35 degrees |
| Minimum gap | 0.50 m beyond both radii |
| Guaranteed direct-path blockers | 2 |
| Planner radius | `0.5 * hypot(width, depth)` |

Each USD obstacle is a `Building_NNN` Xform with a collision-enabled `Body`,
four facade directions of non-colliding decorative windows, and a
non-colliding roof crown plus optional antenna. Facade/window colors, on/off
window pattern, roof style, dimensions, and yaw follow the canonical generator.
Only one obstacle record and radius per building reaches the current planner.

The existing Phase 10B `blocked_goal` safe-failure fixture remains available
for regression only; it is not used by normal Phase 10C QA scenes.

## Exact legacy lighting

All lights live under `/World/GeneratedEpisode/Lights`, so the existing episode
cleanup removes them. The three reference runs used only these exact values:

| Light | Intensity | Exposure / angle | Rotation | Color |
|---|---:|---|---|---|
| Dome | 300 | exposure 0 | n/a | (0.92, 0.96, 1.0) |
| Key | 1300 | angle 4 deg | (315, 0, 35) | (1.0, 0.96, 0.90) |
| Fill | 650 | angle 6 deg | (300, 0, 215) | (0.84, 0.91, 1.0) |

No candidate lighting was generated because exact legacy lighting produced
recognizable edges and low clipping in the current renderer. Across all 18 RGB
QA images, the observed dark-pixel fraction was at most 3.14% and the
overexposed-pixel fraction was below 0.11%.

## Recovered effective camera contract

The FPV camera uses BODY_AXIS +X, 0.45 m forward offset, 0.12 m height, 3.5 m
look-ahead, the manager's effective -0.8 m look-down, focal length 12, and
horizontal aperture 28. Its position is applied as a rigid body mount without
world-space smoothing. The first exact-smoothing QA exposed the UAV body in
mid-flight and near-goal frames: the current bridge updated the smoothed eye at
the 5 Hz publish cadence while aiming at the unsmoothed current pose/yaw. The
resulting eye/target lag put the UAV between the camera and target. Disabling
FPV position smoothing fixes that defect without changing camera geometry.

The Observer uses the manager's effective TOP mode, 9.0 m height, 0.0 m look-at
height, focal length 18, horizontal aperture 22, and smoothing 0.18. It is named
`observer_rgb`, not `top_rgb`, to avoid conflating geometry in metadata.

The interfaces remain:

```text
/uav/isaac/fpv/image/compressed       320x180 JPEG quality 85, target 5 Hz
/uav/isaac/observer/image/compressed  320x180 JPEG, target 2 Hz
/uav/isaac/fpv/depth/compressed       PNG uint16 millimetres, target 5 Hz
```

Observer availability stays auxiliary and cannot invalidate a BC V1 primary
sample. The batch validator accepts the existing Phase 10B pilot's old
`top_rgb` columns as well as the corrected `observer_rgb` name.

## Three-flight visual QA

The bounded command starts a fresh managed XRCE + Isaac/Pegasus/PX4 runtime for
each required seed, applies its scene after the existing safe reset checks,
runs the guarded ASTAR_EXPERT mission, and waits through controlled landing.
The capture node is read-only and records start, mid-flight, and near-goal
frames; it has no control publisher.

```bash
cd /home/noel_614420090/uav-project
UAV_OFFLINE_TIMEOUT_SECONDS=180 ./uav phase10c-visual-qa
./uav phase10c-visual-qa-check \
  --root artifacts/visual_qa/phase10c_highrise_rigid_fpv
```

Each seed directory contains three FPV JPEGs, three Observer JPEGs, three raw
uint16 depth PNGs, three normalized depth previews, scene/path/camera metadata,
complete flight evidence, and a visual contact sheet. These are visual-QA
artifacts, not dataset episodes.

| Seed | Path points | Path / direct m | Detour ratio | Goal hold s | Complete s |
|---:|---:|---:|---:|---:|---:|
| 102001 | 101 | 7.955 / 5.594 | 1.422 | 31.679 | 38.179 |
| 102002 | 77 | 6.055 / 5.592 | 1.083 | 26.111 | 32.561 |
| 102003 | 77 | 6.053 / 5.595 | 1.082 | 30.620 | 37.070 |

Automated validation confirms exactly eight decorated high-rises and two
guaranteed blockers in every distinct layout, meaningful A* detours, the exact
lighting/camera contracts, three valid image/depth phases, and successful safe
terminal state. Every flight reached the goal, landed, disarmed, retained
`failsafe=false`, and recorded zero stream faults.

Manual inspection shows recognizable building boundaries and facade windows in
all start FPV frames, obstacle/building context during avoidance, distinct
layouts, and useful TOP relationships over the three capture phases. After the
rigid-mount correction, none of the nine FPV start/mid/near frames contains the
UAV body. The rejected exact-smoothing artifact is retained separately at
`artifacts/visual_qa/phase10c_canonical_highrise` for A/B comparison.

## Exact legacy versus current integration

| Concern | Canonical legacy | Current integration |
|---|---|---|
| Scene algorithm | Global seeded high-rise generator | Same call order and sampler using local seeded RNG |
| Building rendering | Body/windows/roof hierarchy | Same hierarchy under current episode root |
| Collision | Body only | Body only; decorations excluded |
| Planner | Legacy pipeline's consumers | Current A* receives one radius per building |
| Lighting | Generator-owned exact values | Exact values under cleanup-owned light root |
| FPV | Smoothed dual camera controller | Same geometry and manager overrides; rigid mount removes current-runtime eye/target lag |
| Observer | Controller with runtime override | Effective TOP override, published as Observer |
| Flight/control | Legacy episode pipeline | Unchanged current ROS 2/PX4 Phase 9/10B stack |

## Validation commands

```bash
./uav test
./uav verify
./uav ml-test
./uav phase10b-dataset-check --dataset artifacts/datasets/bc_expert_v1
./uav phase10c-visual-qa-check \
  --root artifacts/visual_qa/phase10c_highrise_rigid_fpv
git diff --check
```

Observed results for this uncommitted review state:

- `./uav test`: 286 tests, 0 failures, 0 skipped.
- `./uav verify`: success; build, tests, interfaces, imports, generated-file
  hygiene, and safety checks passed.
- `./uav ml-test`: 14 tests passed.
- Phase 10B pilot validation: 10 episodes, 333 accepted samples, BC 72D and
  target 3D rebuilt, and all failure episodes safe.
- Phase 10C validator: three seeds valid, all layouts unique, all flights safe.
- `git diff --check`: passed.

Phase 10C stops after these three visual-QA flights pending human confirmation.
It does not authorize dataset collection, training, learned-policy runtime,
object detection, or thermal sensing.
