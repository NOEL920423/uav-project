#!/usr/bin/env bash
set -euo pipefail

TMUX_SESSION="uav"
ROS_PANE="uav:2.6"
TRAIN_PANE="uav:2.5"
ISAAC_PANE="uav:3.0"
ROS_WORKSPACE="/home/noel_614420090/uav_ros2_ws"
ISAAC_ROOT="/home/noel_614420090/isaacsim/_build/linux-x86_64/release"
BOOTSTRAP="/home/noel_614420090/uav-project/isaac/runtime/bootstrap.py"
KIT_PATTERN="^${ISAAC_ROOT}/kit/kit .*isaacsim.exp.full.streaming.kit"

source_ros() {
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source "${ROS_WORKSPACE}/install/setup.bash"
    set -u
}

require_tmux() {
    if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "tmux session '${TMUX_SESSION}' does not exist." >&2
        exit 1
    fi
}

service_call() {
    source_ros
    ros2 service call "$1" std_srvs/srv/Trigger '{}'
}

start_ros() {
    require_tmux
    if pgrep -f '/opt/ros/jazzy/bin/ros2 launch uav_px4_control astar_path_mission.launch.py' >/dev/null; then
        echo "ROS flight launch is already running."
        return
    fi
    if [[ "$(tmux display-message -p -t "${ROS_PANE}" '#{pane_current_command}')" != "bash" ]]; then
        echo "${ROS_PANE} is busy; refusing to overwrite its command." >&2
        exit 1
    fi
    tmux send-keys -t "${ROS_PANE}" \
        "source /opt/ros/jazzy/setup.bash && source ${ROS_WORKSPACE}/install/setup.bash && ros2 launch uav_px4_control astar_path_mission.launch.py" Enter
    echo "ROS flight launch sent to ${ROS_PANE}."
}

wait_for_isaac_manager() {
    source_ros
    for attempt in $(seq 1 30); do
        output=$(timeout 3 ros2 service call /uav_sim/get_status std_srvs/srv/Trigger '{}' 2>&1 || true)
        if grep -q 'status=ready' <<<"${output}"; then
            echo "Isaac UAV manager is ready (attempt ${attempt})."
            return
        fi
        sleep 3
    done
    echo "Timed out waiting for /uav_sim/get_status." >&2
    exit 1
}

start_isaac() {
    require_tmux
    if pgrep -f "${KIT_PATTERN}" >/dev/null; then
        echo "Isaac Kit is already running."
        return
    fi
    if [[ "$(tmux display-message -p -t "${ISAAC_PANE}" '#{pane_current_command}')" != "bash" ]]; then
        echo "${ISAAC_PANE} is busy; refusing to overwrite its command." >&2
        exit 1
    fi
    tmux send-keys -t "${ISAAC_PANE}" \
        "cd ${ISAAC_ROOT} && env -u DISPLAY -u WAYLAND_DISPLAY ./isaac-sim.streaming.sh --livestream 2 --exec ${BOOTSTRAP}" Enter
    echo "Isaac/Pegasus/PX4 bootstrap sent to ${ISAAC_PANE}."
    wait_for_isaac_manager
}

mission_is_terminal() {
    source_ros
    output=$(timeout 6 ros2 service call /uav_mission/get_status std_srvs/srv/Trigger '{}' 2>&1 || true)
    if grep -Eq 'state=(IDLE|COMPLETE|FAILED)' <<<"${output}"; then
        return 0
    fi
    echo "Mission is not confirmed terminal; refusing to stop Isaac." >&2
    echo "${output}" >&2
    return 1
}

stop_isaac() {
    require_tmux
    mapfile -t kit_pids < <(pgrep -f "${KIT_PATTERN}" || true)
    if [[ "${#kit_pids[@]}" -eq 0 ]]; then
        echo "Isaac Kit is not running."
        return
    fi
    if [[ "${#kit_pids[@]}" -ne 1 ]]; then
        echo "Expected one Isaac Kit process, found: ${kit_pids[*]}" >&2
        exit 1
    fi
    mission_is_terminal

    tmux send-keys -t "${ISAAC_PANE}" C-c
    for _attempt in $(seq 1 10); do
        if ! pgrep -f '/PX4-Autopilot/build/px4 ' >/dev/null; then
            break
        fi
        sleep 1
    done
    if pgrep -f '/PX4-Autopilot/build/px4 ' >/dev/null; then
        echo "PX4 did not exit; refusing to terminate Isaac Kit." >&2
        exit 1
    fi

    pid="${kit_pids[0]}"
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}"
        for _attempt in $(seq 1 5); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
    fi
    if kill -0 "${pid}" 2>/dev/null; then
        command_line=$(ps -p "${pid}" -o args=)
        if [[ "${command_line}" == "${ISAAC_ROOT}/kit/kit "*"isaacsim.exp.full.streaming.kit"* ]]; then
            kill -KILL "${pid}"
        else
            echo "PID ${pid} no longer belongs to the expected Isaac process." >&2
            exit 1
        fi
    fi
    echo "Isaac/PX4 stopped after a terminal mission state."
}

start_flight_stack() {
    start_ros
    start_isaac
}

restart_isaac() {
    stop_isaac
    sleep 2
    start_isaac
}

run_episode() {
    service_call /uav_mission/run_episode
}

flight_status() {
    service_call /uav_mission/get_status
    service_call /uav_sim/get_status
}

wait_episode() {
    source_ros
    while true; do
        output=$(ros2 service call /uav_mission/get_status std_srvs/srv/Trigger '{}' 2>&1)
        message=$(grep -o "message='[^']*'" <<<"${output}" | tail -1)
        echo "${message}"
        if grep -Eq 'state=(COMPLETE|FAILED)' <<<"${message}"; then
            return
        fi
        sleep 5
    done
}

start_training_manager() {
    require_tmux
    source_ros
    if ros2 service list 2>/dev/null | grep -qx '/uav_bc/train'; then
        echo "BC training manager is already running."
        return
    fi
    if [[ "$(tmux display-message -p -t "${TRAIN_PANE}" '#{pane_current_command}')" != "bash" ]]; then
        echo "${TRAIN_PANE} is busy; refusing to overwrite its command." >&2
        exit 1
    fi
    tmux send-keys -t "${TRAIN_PANE}" \
        "source /opt/ros/jazzy/setup.bash && source ${ROS_WORKSPACE}/install/setup.bash && ros2 launch uav_px4_control bc_training.launch.py epochs:=30 batch_size:=16" Enter
    for _attempt in $(seq 1 10); do
        if ros2 service list 2>/dev/null | grep -qx '/uav_bc/train'; then
            echo "BC training manager is ready."
            return
        fi
        sleep 1
    done
    echo "Timed out waiting for /uav_bc/train." >&2
    exit 1
}

train_bc() {
    if pgrep -f "${KIT_PATTERN}" >/dev/null; then
        echo "Stop Isaac first to release GPU memory: $0 stop-isaac" >&2
        exit 1
    fi
    start_training_manager
    service_call /uav_bc/train
}

training_status() {
    service_call /uav_bc/get_status
}

list_data() {
    find /home/noel_614420090/uav-project/uav_vision_dataset \
        -maxdepth 1 -type d -name 'dual_camera_episode_bc_astar_*' -printf '%f\n' | sort
    find /home/noel_614420090/uav-project/uav_bc_models \
        -maxdepth 1 -type d -name 'bc_*' -printf '%f\n' | sort
}

usage() {
    cat <<'EOF'
Usage: ~/uav-project/legacy/pipeline/uav_pipeline.sh COMMAND

Flight commands:
  start-flight    Start ROS flight nodes and automatic Isaac/Pegasus/PX4 bootstrap
  run             Request one ROS 2 scene/A*/flight/capture episode
  wait            Monitor until the current episode is COMPLETE or FAILED
  status          Show mission and Isaac manager status
  restart-isaac   Safe terminal-state reset before a new independent episode
  stop-isaac      Stop Isaac/PX4 only after IDLE/COMPLETE/FAILED is confirmed

BC commands:
  train-manager   Start the ROS 2 BC training service manager
  train           Require Isaac stopped, then start training through /uav_bc/train
  train-status    Show ROS 2 BC training state
  list-data       List corrected episodes and trained model runs
EOF
}

case "${1:-help}" in
    start-flight) start_flight_stack ;;
    run) run_episode ;;
    wait) wait_episode ;;
    status) flight_status ;;
    restart-isaac) restart_isaac ;;
    stop-isaac) stop_isaac ;;
    train-manager) start_training_manager ;;
    train) train_bc ;;
    train-status) training_status ;;
    list-data) list_data ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
