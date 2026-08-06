# ROS 2 developer commands

## Purpose and safety boundary

The repository-root `./uav` wrapper replaces repeated environment, build, test,
inspection, and offline checks with one reproducible entry point. It is a
non-flight developer tool: it does not start Isaac Sim, Pegasus, PX4, Micro
XRCE-DDS, cameras, recorders, OFFBOARD mode, arming, or takeoff. Phase 6 starts
only synthetic candidate publishers, the offline mux, existing follower and
fixed-step plant as required; it does not publish `/fmu/in/*` commands.

Run `./uav help` for the concise command list.

## Isolated environment

ROS commands run through `env -i`, retaining only `HOME`, `USER`,
`PATH=/usr/bin:/bin`, `LANG=C.UTF-8`, optional `TERM`, and the offline timeout.
The wrapper sources `/opt/ros/jazzy/setup.bash`, verifies `/usr/bin/python3`, and
sources only this repository's `ros2_ws/install/setup.bash` when required. It
never sources the legacy `uav_ros2_ws` or inherits virtualenv, pyenv, AMENT, or
COLCON overlays.

Every clean command prints the selected Python, Python version, ROS distribution,
workspace path, and overlay state.

## Commands

| Command | Behavior |
|---|---|
| `./uav status` | Read-only Git and workspace-directory status |
| `./uav doctor` | Dependency, contamination, branch, overlay, and topic checks |
| `./uav build` | Clean Jazzy `colcon build --symlink-install` |
| `./uav test` | Clean sourced `colcon test` and verbose results |
| `./uav verify` | Doctor, build, tests, packages, interfaces, imports, and scans |
| `./uav offline` | Interactive offline planner launch without a default timeout |
| `./uav offline-check` | Finite, logged, deterministic offline validation |
| `./uav topics` | ROS graph listing and `/fmu/in/*` rejection |
| `./uav interfaces` | Custom definitions and canonical ROS topic/service names |
| `./uav mux-check` | Nominal four-source arbitration and HOLD handoffs |
| `./uav mux-safety-check` | Stale-source latch, explicit recovery, internal HOLD |
| `./uav control-stack-check` | Scene through follower, mux and offline plant |
| `./uav shell` | A new interactive clean shell with a `[uav-ros2]` prompt |

Typical daily use:

```bash
./uav doctor
./uav build
./uav test
./uav verify
./uav offline-check
./uav trajectory-check
./uav pipeline-check
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
```

## Offline modes and launch arguments

`./uav offline` hands the terminal directly to `ros2 launch`; it has no timeout
and may be stopped with Ctrl+C. `./uav offline-check` uses the Phase 3 harness,
checks its structured success marker, source, validity, path counts/frame, and
writes a timestamped ignored log under `run_logs/`. Its default safety timeout
is 20 s:

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=30 ./uav offline-check
```

Arguments after either subcommand are passed to the launch file:

```bash
./uav offline use_sim_time:=false
./uav offline-check enable_bspline:=true fixture:=bspline-safe-single-obstacle
./uav offline-check enable_bspline:=true fixture:=bspline-rejected-corner-cut
./uav offline-check enable_bspline:=false fixture:=bspline-disabled
```

Phase 4 adds two finite non-flight checks. `trajectory-check` publishes a
validated `px4_ned` path directly to the parameterizer; `pipeline-check`
publishes a fixed scene and verifies the complete A* / B-spline-or-fallback /
timed-trajectory chain. Both run in the same clean Jazzy environment, forward
launch arguments, scan for `/fmu/in/*`, and write timestamped ignored logs:

```bash
./uav trajectory-check fixture:=straight-line
./uav trajectory-check fixture:=impossible-config-rejection
./uav trajectory-check fixture:=wrong-frame
./uav pipeline-check enable_bspline:=true
./uav pipeline-check enable_bspline:=false
```

The standalone fixture vocabulary is `straight-line`, `phase3-bspline`,
`sharp-bend`, `high-curvature`, `duplicate-adjacent`, `two-point`,
`invalid-one-point`, `nonfinite`, `yaw-wrap`, `jerk-scaling`,
`impossible-config-rejection`, and `wrong-frame`.

Available edge fixtures are `short-two-point-path`, `three-point-path`,
`duplicate-control-point-path`, `self-intersection-candidate`, and
`curvature-limit-rejection`. To reproduce the six-scene geometric report after
building:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run uav_navigation geometric_path_comparison
```

## Phase 5 offline tracking checks

Phase 5 remains non-flight. Its follower publishes only the ROS-level
`/uav/control/astar_command` candidate and reference/status topics in
`px4_ned`; the launch files contain no PX4, Isaac Sim, Pegasus, XRCE, arming,
OFFBOARD, or `/fmu/in/*` path.

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
```

`tracking-check` runs a fixed timed trajectory through the follower and the
deterministic first-order plant until `GOAL_HOLD`. `tracking-safety-check`
defaults to `stale-odometry` and requires an exact zero HOLD with the expected
reason. `full-pipeline-check` validates the complete fixed scene -> A* ->
accepted B-spline or A* fallback -> timed trajectory -> follower -> plant
chain. All three commands are finite, scan for `/fmu/in/*`, and write ignored
logs under `run_logs/`.

Useful safety fixtures include:

```bash
./uav tracking-safety-check fixture:=stale-odometry
./uav tracking-safety-check fixture:=invalid-validity-flag
./uav tracking-safety-check fixture:=wrong-odometry-frame
./uav tracking-safety-check fixture:=excessive-tracking-error
```

The complete deterministic vocabulary is `straight-trajectory`,
`phase3-bspline-accepted`, `astar-fallback`, `sharp-dynamically-valid`,
`start-position-offset`, `constant-horizontal-disturbance`,
`duplicate-trajectory-message`, `stale-odometry`,
`stale-trajectory-validity`, `invalid-validity-flag`,
`wrong-odometry-frame`, `nonfinite-odometry`, `backward-time-jump`,
`command-speed-saturation`, `command-acceleration-saturation`,
`excessive-tracking-error`, `successful-goal-settling`,
`terminal-not-reached`, `yaw-wrap-crossing`, and
`invalid-command-rejection`.

To reproduce the pure eight-fixture table after building:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run uav_navigation tracking_comparison
```

## Phase 6 offline mux checks

Phase 6 owns only ROS-level arbitration. `mux-check` requests
`ASTAR_EXPERT -> HUMAN_JOYSTICK -> NAVRL_POLICY -> HOLD`, requires an
exact-zero HOLD barrier between movement sources, and verifies one selected
publisher. `mux-safety-check` forces selected-source staleness, checks the
latched HOLD cannot auto-recover, then performs an explicit recovery request;
it also verifies an invalid external HOLD cannot disable internal HOLD.
`control-stack-check` connects the real Phase 5 follower to the mux and makes
the offline plant consume only `/uav/control/selected_command`.

```bash
./uav mux-check
./uav mux-safety-check
./uav control-stack-check enable_bspline:=true
```

All commands are finite, write ignored logs under `run_logs/`, scan the live
graph for `/fmu/in/*`, and do not start PX4, Isaac Sim, joystick hardware or a
NavRL runtime/model.

## Why shell creates a new shell

An executed script cannot safely change its parent shell's environment.
`./uav shell` therefore uses `exec` to open a child Bash process with Jazzy and,
when present, this workspace overlay sourced. Ctrl+D exits cleanly and returns
to the original terminal without contaminating it.

The startup summary prints `ROS_DISTRO`, Python, workspace, overlay state, and
the `px4_ned` planning frame.

## Troubleshooting

If pyenv or a virtual environment selects an incompatible Python, do not install
ROS dependencies into it. Run `./uav doctor`; clean commands intentionally
discard pyenv shims, `PYENV_VERSION`, `VIRTUAL_ENV`, and inherited prefix paths.

If `install/setup.bash` is missing, run:

```bash
./uav build
```

The wrapper never deletes `build/`, `install/`, or `log/`. Remove or relocate
generated artifacts manually only when that separate action is explicitly
intended.

## Optional alias

The wrapper does not modify shell startup files. If desired, manually add:

```bash
alias uav="$HOME/uav-project/uav"
```

Then a new shell may use `uav verify`, `uav offline`, and `uav shell`.
