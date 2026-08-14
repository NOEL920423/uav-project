# ML artifact inventory

Audit date: 2026-08-14 (Asia/Taipei)

Generated datasets, checkpoints, training runs, and legacy image collections
remain ignored by Git. This file preserves their reproducible source paths,
summary checksums, and validation result without committing large binary data.

## Legacy RGB collection

- 29/29 structurally valid episodes
- 5,293 paired FPV/TOP frames; 10,586 decodable 960x540 RGB PNGs
- 27 A* expert episodes and 2 policy rollouts
- 0 corrupt, missing, orphan, dimension-mismatched, or duplicate-adjacent images
- 4,033,397,516 source bytes

The audit report is generated under `uav_vision_dataset/_audit/` by
`./uav rgb-audit`.

## Isaac city dataset

`datasets/isaac_city_bc_v0/` contains 24 train, 6 validation, and 6 test
episodes with 1,768, 447, and 434 synchronized samples. Every episode completed
with the A* expert. Every stored array was checked for its versioned shape,
finite state/action values, and normalized action bounds.

Summary checksums:

```text
ff1b5809af0d6eb6f7c61c7f30b2b01c8aac4fc8523c54a2fba7c725c99604e0  datasets/isaac_city_bc_v0/metadata.json
aa01bc9204268150010a5b2cb94d84cfca56e06f622a9d06e5e441b1a642a81f  datasets/isaac_city_bc_v0_latent/metadata.json
```

## Reload-verified model artifacts

The Autoencoder, latent BC, PPO 50k, and PPO 100k checkpoints were loaded with
PyTorch on the audit date. Their source summaries have these checksums:

```text
d62148a2ea6d88995871fa2a45b5708a03d59879d8359d54afb8542506215589  autoencoder_runs/rgb_ae_v0_baseline_20260811/summary.json
e06097a09bde75ee7cf38db0e38b893b1556284467eda8c4605b94092d2e1958  training_runs/latent_bc_city_v0/summary.json
b467f61dd20acb26b1eb962023a8746aeb58a3fa15a50d33f54e812c31cbe643  training_runs/ppo_city_50k_20260811_185451/summary.json
28d8e813908f5c1238c55797c3687ab443546d035c3ee4fd50f904999310aef9  training_runs/ppo_city_100k_20260811_193952/summary.json
```

These checksums inventory the current local evidence. They are not a remote
backup of the ignored binary artifacts.
