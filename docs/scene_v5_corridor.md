# v5 shared obstacle corridor

Select `--scene v5` to generate two cylinders whose centers share a corridor
from the start `(0, 0)` to the target `(3, 5)`. The corridor is **1.0 m wide
in total**, with a signed perpendicular center offset in `[-0.5, 0.5] m`.
Both obstacles sample the full segment; neither has a separate longitudinal
interval. The cylinder body may extend outside the corridor.

`config/scenes/v5.json` controls the total width through
`obstacle_placement.width_m`. The existing general random size limits remain:

- Radius basis width and depth: independently sampled in `[0.46, 0.72] m`.
- Cylinder radius: `0.5 * hypot(radius_basis_width, radius_basis_depth)`.
- Height: `[2.8, 5.2] m`; yaw: `[-35, 35]` degrees.

Both cylinders use these random limits; there is no fixed special-large
cylinder. Start and target clear radii remain 1.5 m, obstacle surface separation
remains at least 0.50 m, and complete cylinders remain inside the arena walls.
These constraints reject otherwise valid center samples. A center in the
corridor does not necessarily block the direct path.

The generator samples a complete layout again if either obstacle cannot be
accepted. It fails explicitly after `MAX_PLACEMENT_ATTEMPTS` layout attempts.
The same profile and episode seed reproduce dimensions, positions, and details.
The profile, hash, and corridor placement contract are stored in scene metadata
and collection/evaluation snapshots.

From the repository root in the remote Ubuntu VS Code terminal:

```bash
# Collect into a new dataset.
./uav expert-collect --scene v5 --episodes 100 --dataset cylinder_v5_new

# Evaluate an existing checkpoint in the new scene.
./uav bc-eval --scene v5 --checkpoint /path/to/best.pt --episodes 1 --visible

# Optional offline fixture; does not start Isaac, PX4, or a real collection.
./uav expert-collect --scene v5 --episodes 2 --dry-run
```

The default remains v4. The v3/v4 profiles and their historical seed contracts
are preserved. Use a new dataset for v5; do not resume a v3/v4 collection with
different scene parameters. Flight success in v5 requires separate closed-loop
validation.
