# Phase 10A: dataset contract and one automated expert episode

## Boundary

Phase 10A defines `bc_expert_v1.0` and records exactly one accepted
`ASTAR_EXPERT` episode. It reuses the Phase 9 planner, follower, mux, output
gate, PX4 streamer, flight supervisor, Isaac scene bridge, and acceptance
monitor. It does not add multi-episode collection, training, learned-policy
runtime, PPO, object detection, or reward changes.

## Audited model and preprocessing contract

- Source image: FPV RGB, stored as 320x180 JPEG quality 85.
- Encoder input: RGB `[B,3,72,128]` float32 in `[0,1]`.
- Preprocessing: PIL RGB conversion, bilinear resize to 128x72, no crop, HWC
  to CHW, divide uint8 values by 255. There is no ImageNet normalization.
- Latent: four stride-2 convolutions produce `128x5x8`; a linear layer
  produces an unbounded 64D vector.
- State: body `[forward,right]` velocity in m/s; body-frame horizontal unit
  goal direction; `min(max(goal_distance_m / 10.0, 0), 1)`; and the immediately
  preceding applied normalized action. A new action history starts at zero;
  this flight dataset accepts only samples with an actual preceding command.
- Observation: `64D latent + 8D state = 72D`. Actor normalization remains the
  existing training-only, per-dimension mean/std stored with the BC checkpoint.
- Target: normalized body `[v_forward,v_right,yaw_rate]` in `[-1,1]`, using
  physical limits `[1.0 m/s, 0.8 m/s, 1.0 rad/s]`. The source Phase 8 physical
  body command is also retained so clipping does not discard source truth.

For NED yaw `psi`, horizontal vectors are converted with:

```text
forward =  cos(psi) * north + sin(psi) * east
right   = -sin(psi) * north + cos(psi) * east
```

The older depth/4D `bc_v0.1` contract remains in the repository but is not the
contract used here.

## Synchronization and storage contract

The Isaac bridge publishes one 320x180 compressed FPV JPEG approximately every
0.2 seconds. Each image header timestamp is the join anchor. The recorder waits
100 ms, then selects the nearest ROS timestamp from PX4 NED odometry, A* expert
command, mux status, and flight status. State and action absolute timestamp
errors must each be at most 100 ms. The mission goal is a durable static scene
sample; its Isaac `[x=east,y=north]` coordinates are explicitly mapped to NED.

Only samples with `TRACKING`, valid `ASTAR_EXPERT` mux selection, a valid
follower command, and a preceding expert command are accepted. Missing, stale,
over-tolerance, HOLD, or non-tracking anchors are counted by reason and never
written to `samples.csv`. Mission phase transitions are retained in
`episode.json`, so takeoff, goal, landing, and completion remain auditable.

Generated data is Git-ignored:

```text
artifacts/datasets/bc_expert_v1/
├── dataset_manifest.json
└── episode_000001/
    ├── images/frame_*.jpg
    ├── samples.csv
    ├── episode.json
    └── validation.json
```

Every accepted CSV row contains episode/sample/image identifiers and timestamp,
state/action timestamps and join errors, NED pose/yaw, body velocity, body goal
direction, raw/normalized distance, physical and normalized current/previous
actions, mission phase, and final success/failure fields.

## Reproduction

Terminal 1:

```bash
MicroXRCEAgent udp4 -p 8888
```

Terminal 2, from the Isaac Sim release directory:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY UAV_PHASE10A_CAMERA=1 \
  ./isaac-sim.streaming.sh --livestream 2 \
  --exec /home/noel_614420090/uav-project/isaac/runtime/bootstrap.py
```

Terminal 3:

```bash
cd /home/noel_614420090/uav-project
UAV_OFFLINE_TIMEOUT_SECONDS=180 ./uav phase10a-expert-episode
./uav phase10a-dataset-check
```

The command refuses to overwrite an existing V1 root, verifies a live JPEG
topic before authorizing flight, runs the unchanged Phase 9 acceptance launch
with the recorder enabled, and validates the dataset with the frozen checkpoint.

## Accepted runtime evidence

Accepted flight evidence:
`run_logs/px4-sitl-flight_20260815T132951Z.json` (generated and Git-ignored).

| Evidence | First observed (s) |
|---|---:|
| A* path, B-spline, timed trajectory valid | 5.053 |
| ASTAR_EXPERT selected | 5.103 |
| stable 20 Hz stream | 7.353 |
| PX4 OFFBOARD | 7.372 |
| PX4 armed | 7.418 |
| Isaac takeoff | 14.885 |
| trajectory motion | 17.633 |
| Isaac goal tolerance | 19.726 |
| supervisor goal reached / landing command | 21.703 |
| PX4 and Isaac landed | 25.762 |
| final disarm / COMPLETE | 27.910 |

Flight statistics: maximum PX4 altitude 1.541 m, maximum Isaac altitude
1.547 m, minimum PX4 goal distance 0.018 m, minimum Isaac goal distance
0.106 m, maximum verified stream rate 20.011 Hz, zero recorded stream faults,
and final Isaac pose `[0.513, 3.176, 0.062]` m.

Dataset validation: 27 accepted tracking samples, 90 rejected non-training
anchors (16 before goal availability, 2 mux transition, 72 non-tracking),
4.327 Hz observed sample rate, strictly monotonic image/state/action timestamps,
maximum image/state error 3.319 ms, maximum image/action error 9.889 ms,
27 readable 320x180 JPEGs, 64D latent rebuild, 72D observation rebuild, 3D
target rebuild, and approximately 231 kB total disk use.

## Regression results

```text
./uav test                       286 tests, 0 failures
./uav verify                     SUCCESS (build/tests/interfaces/import/safety)
./uav ml-test                    14 tests, OK
runtime contract pytest          5 passed
./uav phase10a-dataset-check     valid=true
git diff --check                 clean
```

The first `./uav test` was accidentally run while the live Isaac ROS graph was
still active and one navigation integration assertion saw cross-test traffic.
After stopping Isaac, PX4, XRCE, and the ROS daemon, the isolated full suite and
the complete `./uav verify` rerun both passed 286/286. No algorithm change was
made in response to that environment-contamination failure.

## Acceptance result

- One accepted automated expert episode: pass.
- Actual A* expert takeoff, trajectory following, goal, controlled landing,
  and disarm: pass from both PX4 and Isaac state evidence.
- JPEG open/resolution/visual-range checks: pass, 27/27.
- Sampling rate and monotonic timestamps: pass.
- Image/state/action synchronization within 100 ms: pass.
- Required fields and correct final metadata: pass.
- Frozen encoder 64D, complete 72D observation, and 3D target rebuild: pass.
- Phase 9/9.5 regressions and fail-closed architecture: pass.

An initial diagnostic attempt placed the camera inside the Iris body. Its 26
frames were rejected after visual inspection, retained only under ignored
`artifacts/rejected/phase10a_camera_inside_body_20260815T132558Z`, and are not
part of the V1 dataset. The accepted camera uses the previously proven 0.45 m
forward / 0.12 m upward mount and the validator now rejects blank/dark frames.

## Remaining risks

- This proves one deterministic, sparse Phase 9 scene only; it says nothing
  about dataset diversity or learned-policy generalization.
- Isaac streaming load produced 4.33 Hz rather than exactly 5 Hz. The contract
  target remains 5 Hz and timestamp joins, not assumed cadence, determine
  validity.
- The fixed scene is visually dark and sparse, although every accepted frame
  passes open/resolution/dynamic-range checks. Scene diversity and exposure are
  Phase 10B concerns and were intentionally not changed here.
- The existing autoencoder was trained on older imagery, so its 64D output is
  reconstructible but its usefulness on this domain is not established.
