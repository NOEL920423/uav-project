# uav_navigation

Future owner of path planning, smoothing, tracking, and candidate generation.

- Inputs: typed scene state, pose, twist, camera data, and mission state.
- Outputs: raw/smoothed paths, planning status, and candidate commands (future).
- Must not: publish `/fmu/in/*`, own simulator lifecycle, or record datasets.
- Phase 1 behavior: an idle node with planning disabled by default.
- Future phase: add pure planning adapters only after frame contracts are closed.
