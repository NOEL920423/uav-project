# Future generated-artifact layout

This directory only records a possible future organization for generated,
gitignored data:

```text
artifacts/
├── datasets/
├── checkpoints/
├── runs/
├── logs/
└── media/
```

Phase 9.5 does not move existing datasets, checkpoints, training or
autoencoder runs, runtime logs, videos, rosbags, pose logs, or model outputs.
Their current paths and all ML defaults remain unchanged.
