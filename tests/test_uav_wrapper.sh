#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UAV="$REPO_ROOT/uav"
PPO_RUNNER="$REPO_ROOT/train_ppo.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

[[ -x "$UAV" ]] || fail "$UAV is not executable"
[[ -x "$PPO_RUNNER" ]] || fail "$PPO_RUNNER is not executable"
/usr/bin/bash -n "$UAV"
/usr/bin/bash -n "$PPO_RUNNER"
ppo_help="$($PPO_RUNNER --help)"
[[ "$ppo_help" == *"./train_ppo.sh 50k"* ]] || \
    fail "PPO runner help is missing the 50k preset"

"$UAV" help >/dev/null
if "$UAV" definitely-not-a-command >/dev/null 2>&1; then
    fail "unknown command unexpectedly returned zero"
fi

before_status="$(git -C "$REPO_ROOT" status --porcelain=v1)"
"$UAV" status >/dev/null
after_status="$(git -C "$REPO_ROOT" status --porcelain=v1)"
[[ "$before_status" == "$after_status" ]] || fail "status modified the repository"

doctor_output="$("$UAV" doctor 2>&1)"
[[ "$doctor_output" == *"Python path: /usr/bin/python3"* ]] || \
    fail "doctor did not select /usr/bin/python3"
[[ "$doctor_output" == *"ROS distribution: jazzy"* ]] || \
    fail "doctor did not source ROS 2 Jazzy"

temporary_root="$(mktemp -d /tmp/uav-wrapper-test.XXXXXX)"
cp "$UAV" "$temporary_root/uav"
chmod +x "$temporary_root/uav"
set +e
bad_output="$(cd "$temporary_root" && ./uav status 2>&1)"
bad_status=$?
set -e
((bad_status != 0)) || fail "incorrect repository assumptions were accepted"
[[ "$bad_output" == *"repository check failed"* ]] || \
    fail "incorrect repository error was unclear"

if /usr/bin/grep -nE 'source[^\n]*uav_ros2_ws' "$UAV"; then
    fail "wrapper contains a legacy workspace source command"
fi
/usr/bin/grep -Fq \
    'ros2 launch uav_navigation astar_planner_offline.launch.py "$@"' \
    "$UAV" || fail "offline launch arguments are not forwarded"
/usr/bin/grep -Fq 'trajectory_parameterizer_offline.launch.py "$@"' \
    "$UAV" || fail "trajectory-check arguments are not forwarded"
/usr/bin/grep -Fq 'planning_to_trajectory_offline.launch.py "$@"' \
    "$UAV" || fail "pipeline-check arguments are not forwarded"
help_output="$("$UAV" help)"
[[ "$help_output" == *"trajectory-check"* ]] || \
    fail "trajectory-check is missing from help"
[[ "$help_output" == *"pipeline-check"* ]] || \
    fail "pipeline-check is missing from help"

git -C "$REPO_ROOT" check-ignore -q ros2_ws/build/example
git -C "$REPO_ROOT" check-ignore -q ros2_ws/install/example
git -C "$REPO_ROOT" check-ignore -q ros2_ws/log/example
git -C "$REPO_ROOT" check-ignore -q run_logs/example.log

if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$UAV" "$REPO_ROOT/tests/test_uav_wrapper.sh"
fi

printf 'PASS: uav wrapper tests\n'
