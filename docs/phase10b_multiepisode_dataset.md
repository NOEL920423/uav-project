# Phase 10B: multi-episode expert dataset collection

> **Status after Phase 10C:** this ten-episode result is retained as an
> engineering pilot for automation and synchronization only. Its simplified
> scene/camera distribution is not an approved training dataset. Phase 10C
> replaces that runtime distribution with the canonical later high-rise scene,
> exact episode lighting, FPV geometry, and TOP Observer camera; it does not
> retroactively alter this artifact.

## Boundary and result

Phase 10B extends the Phase 10A recorder into an unattended, seeded ten-episode
pilot. It reuses the validated Phase 9 A* -> timed trajectory -> follower ->
ASTAR_EXPERT mux -> safety gate -> PX4 flight stack. It does not change the
encoder, BC/PPO models, normalization, planner, controller, or safety behavior.

The accepted pilot is Git-ignored at
`artifacts/datasets/bc_expert_v1_phase10b`. Its aggregate validator reports ten
unique scenes and seeds, nine successful flights, one intentionally blocked and
safely landed flight, 333 accepted samples, and a fully rebuildable BC V1
contract.

## Architecture and reset contract

`episode_scene.py` is the deterministic, ROS-independent scene generator.
`episode_scene_client` first requires live PX4 evidence for landed, disarmed,
and no failsafe, publishes the episode/seed/mode request, and waits for a
matching Isaac acknowledgement. The embedded Isaac runtime owns only scene and
sensor publication. The external recorder owns timestamp joins and storage.

Each episode uses a full managed restart of Isaac, Pegasus, PX4 SITL, and XRCE.
This is necessary because a persistent Pegasus/PX4 process retains simulated
battery health across flights; merely resetting the vehicle pose or flight mode
does not restore pre-flight health. The wrapper refuses to run alongside an
existing Isaac/PX4/XRCE process, records exact process groups, and uses bounded
SIGINT, SIGTERM, then SIGKILL teardown. It never teleports the vehicle while PX4
is running.

Scene seeds are `101001` through `101010`. Normal scenes vary goal heading and
distance and contain two deterministic obstacles. Episode 5 uses
`blocked_goal`: an additional obstacle covers the goal so mission replanning
fails. The existing supervisor performs controlled landing and the batch
continues only after landed/disarmed/no-failsafe evidence is recorded. A
safety-critical runtime/bootstrap failure stops the batch.

## Dataset and sensor contract

The `bc_expert_v1.0` contract is unchanged:

```text
FPV RGB JPEG -> resize 128x72, RGB float32 [0,1] -> frozen encoder -> 64D
+ body velocity [forward,right] m/s
+ body-frame horizontal goal unit vector [forward,right]
+ clip(raw goal distance / 10 m, 0, 1)
+ immediately preceding normalized 3D expert action
= 72D observation

target = [v_forward / 1.0 m/s, v_right / 0.8 m/s,
          yaw_rate / 1.0 rad/s], clipped to [-1,1]
```

The primary stream remains FPV RGB, 320x180 JPEG quality 85, target 5 Hz.
The image header is the synchronization anchor; the nearest state, expert
action, mux, and flight samples must be within 100 ms. Over-tolerance, stale,
non-tracking, invalid-command, or non-ASTAR_EXPERT anchors are rejected and
counted. Every accepted row retains physical and normalized actions and all
inputs needed to recreate 64D/72D/3D.

Auxiliary streams are deliberately outside the 72D observation:

- TOP RGB: 320x180 JPEG, target 2 Hz, nearest match within 350 ms.
- FPV depth: PNG uint16 millimetres, target 5 Hz, valid range 50--30000 mm,
  clipped to that range, with zero encoding invalid pixels; match within 100 ms.

Auxiliary paths, timestamps, match errors, and availability are written to
`auxiliary.csv`. Missing or over-tolerance auxiliary data never invalidates an
otherwise legal primary BC sample.

Generated layout:

```text
artifacts/datasets/bc_expert_v1_phase10b/
├── dataset_manifest.json
├── batch_validation.json
└── episode_000001/ ... episode_000010/
    ├── images/
    ├── top_rgb/
    ├── fpv_depth/
    ├── samples.csv
    ├── auxiliary.csv
    ├── episode.json
    ├── flight_evidence.json
    └── validation.json
```

## Reproduction

The pilot command manages XRCE and Isaac/Pegasus/PX4 itself. No pre-existing
instance may be running, and the target dataset directory must not exist:

```bash
cd /home/noel_614420090/uav-project
UAV_OFFLINE_TIMEOUT_SECONDS=180 ./uav phase10b-expert-pilot
./uav phase10b-dataset-check \
  --dataset artifacts/datasets/bc_expert_v1_phase10b --episodes 10
```

`UAV_ISAAC_SIM_RELEASE` may override the default Isaac release directory;
`UAV_PHASE10B_BASE_SEED` may override the default base seed of 101000. The
collector refuses overwrite rather than mixing runs.

## Accepted pilot evidence

| Episode | Seed | Mode | Result | Samples | Path (m) |
|---|---:|---|---|---:|---:|
| 000001 | 101001 | normal | success | 29 | 2.765 |
| 000002 | 101002 | normal | success | 42 | 3.736 |
| 000003 | 101003 | normal | success | 43 | 3.544 |
| 000004 | 101004 | normal | success | 31 | 3.256 |
| 000005 | 101005 | blocked_goal | safe failure | 0 | 0.000 |
| 000006 | 101006 | normal | success | 30 | 3.032 |
| 000007 | 101007 | normal | success | 41 | 3.505 |
| 000008 | 101008 | normal | success | 48 | 4.445 |
| 000009 | 101009 | normal | success | 32 | 3.422 |
| 000010 | 101010 | normal | success | 37 | 3.337 |

All nine normal episodes contain observed OFFBOARD, ARM, takeoff, tracking,
goal, AUTO_LAND, landed, and disarm transitions. Episode 5 took off under the
unchanged Phase 9 sequence, rejected the blocked mission path, timed out in
replanning, commanded AUTO_LAND, and ended landed/disarmed with failsafe false.
Thus a flight failure did not hang or corrupt the batch.

Aggregate dataset statistics:

- 333 accepted samples; 1,000 rejected anchors: 823 non-tracking, 157 before
  goal availability, and 20 before stable ASTAR_EXPERT mux selection.
- Effective sampling rate: mean 4.568 Hz, p95 4.618 Hz, maximum 4.622 Hz.
- Image/state join error: mean 2.898 ms, p95 3.260 ms, maximum 4.229 ms.
- Image/action join error: mean 4.986 ms, p95 9.392 ms, maximum 10.020 ms.
- Auxiliary availability: depth 333/333; TOP RGB 222/333, with 111 explicitly
  recorded missing matches and no primary rejection caused by them.
- Total size 11,573,074 bytes; average 1.157 MB/episode; linear estimate about
  1.157 GB per 1,000 episodes for this short-scene pilot.
- The frozen checkpoint SHA-256 is
  `a10eb39abdc4a797dc59523580cc5ab1dcf567fe7aaf1cffd8d0c888ca4349e3`.
  The validator opened every FPV image and rebuilt all 333 latent64,
  observation72, and target3 records.

## Diagnostics and remaining risks

Two pre-pilot camera diagnostics were rejected for a dark horizon before the
camera pitch was corrected. A persistent-runtime diagnostic was also rejected
after it demonstrated retained simulated battery health across episodes. These
ignored runs remain under `artifacts/rejected/` and are not part of the accepted
dataset.

- This is a ten-scene pilot with short paths and simple cylindrical obstacles;
  it does not establish enough diversity or scale for BC training.
- The observed primary rate is consistently near, but below, the 5 Hz target.
  Correct timestamp joins are enforced, but future larger scenes should monitor
  rendering load.
- The existing autoencoder was trained on older imagery. Rebuildability is
  proven; representation quality or policy generalization is not.
- TOP RGB is intentionally sparse relative to FPV and is debug-only. Depth is
  stored efficiently but has not been calibrated for a learned observation.
- Isaac Kit sometimes requires forced teardown after controlled flight cleanup.
  The managed wrapper now bounds this wait and targets only its recorded process
  group; this remains a simulator shutdown nuisance, not a flight safety bypass.

Phase 10B stops here. It does not start BC/PPO training, encoder retraining,
object detection, thermal sensing, or learned-policy flight.

## Regression results

```text
./uav test                         286 tests, 0 failures
./uav verify                       SUCCESS (build/tests/interfaces/import/safety)
./uav ml-test                      14 tests, OK
runtime/scene contract pytest      10 passed
./uav phase10a-dataset-check       valid=true, 27/27 rebuilt
./uav phase10b-dataset-check       valid=true, 333/333 rebuilt
bash -n / py_compile / diff-check  clean
```
