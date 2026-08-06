# ROS 2 developer commands

## Purpose and safety boundary

The repository-root `./uav` wrapper replaces repeated environment, build, test,
inspection, and offline-planner commands with one reproducible entry point. It
is a non-flight developer tool: it does not start Isaac Sim, Pegasus, PX4,
Micro XRCE-DDS, cameras, recorders, controllers, OFFBOARD mode, arming, takeoff,
or B-spline behavior, and it does not publish `/fmu/in/*` commands.

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
| `./uav interfaces` | Five custom definitions and planner topic names |
| `./uav shell` | A new interactive clean shell with a `[uav-ros2]` prompt |

Typical daily use:

```bash
./uav doctor
./uav build
./uav test
./uav verify
./uav offline-check
```

## Offline modes and launch arguments

`./uav offline` hands the terminal directly to `ros2 launch`; it has no timeout
and may be stopped with Ctrl+C. `./uav offline-check` uses the Phase 2 harness,
checks its structured success marker and path counts/frame, and writes a
timestamped ignored log under `run_logs/`. Its default safety timeout is 20 s:

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=30 ./uav offline-check
```

Arguments after either subcommand are passed to the launch file:

```bash
./uav offline use_sim_time:=false
./uav offline-check use_sim_time:=false
```

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
