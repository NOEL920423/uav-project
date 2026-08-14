# Legacy RGB episode audit

## Scope

Run the non-destructive audit with:

```bash
./uav rgb-audit
```

Reports are written under the ignored `uav_vision_dataset/_audit/` directory.
The audit verifies every PNG, manifest row, FPV/TOP pair, frame index, sim-time
sequence, image dimension, episode ID, and matching pose log. It hashes adjacent
frames to detect exact duplicates. Source episodes are never moved or deleted.

## Clean training view

The clean Autoencoder view uses only the onboard-like FPV camera from valid A*
expert episodes. TOP images are external observer views and are excluded. BC
policy rollouts are also excluded so that an unsupervised encoder cannot see
evaluation observations during training.

The split unit is a complete episode, never an individual frame. The fixed seed
is `614420090`; each environment group contributes one validation episode and
one test episode. This prevents adjacent-frame leakage and makes the split
reproducible.

## Current audit result

The 2026-08-11 audit found 29 structurally valid episodes, 5,293 paired frames,
10,586 decodable 960x540 RGB PNGs, and matching pose logs for every episode.
There were no corrupt, missing, orphan, dimension-mismatched, nonsequential, or
exactly repeated adjacent images. No source data was deleted or quarantined.

- 27 A* expert episodes are eligible for the clean FPV view.
- 2 BC policy rollouts are retained separately as evaluation evidence.
- The episode split is 19 train, 4 validation, and 4 test.
- Raw source size is 4,033,397,516 bytes.

## Research limitation

Visual inspection found simulator-only privileged cues in FPV images: the red
goal marker is visible, and some frames show the rendered cyan path/waypoints.
The present RGB collection is therefore suitable for Autoencoder pipeline and
overfitting experiments, but not for claiming map-free visual navigation. A
future recorder must hide planning/debug geometry from the FPV render product
before collecting policy-training data.
