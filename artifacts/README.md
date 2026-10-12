# Generated data locations

Current collection, training, and evaluation outputs live at the repository root:

```text
uav-project/
├── datasets/                 # Expert datasets, images, and collection metadata
├── experiments/
│   ├── autoencoder/          # AE checkpoints and training results
│   └── bc/                   # BC checkpoints and training results
├── evaluations/
│   └── bc/                   # Closed-loop results, videos, plots, and replay
└── artifacts/
    ├── datasets/             # Retained copies of the original datasets
    ├── experiments/
    │   ├── autoencoder/      # Retained original AE results
    │   ├── bc/               # Retained original BC results
    │   └── bc_baseline/      # Legacy results kept in place
    └── ...                   # Other existing diagnostics and regressions
```

All generated data directories are ignored by Git. Only this README is tracked
under `artifacts/`. Short dataset names resolve under `datasets/`; new AE and BC
runs use `experiments/`, and BC evaluations use `evaluations/bc/`.

Original dataset and experiment copies remain available. The latest 15 results
from `artifacts/evaluations/bc_flight/` were moved to `evaluations/bc/` without
renaming, and older evaluation result folders were deleted.
