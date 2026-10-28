#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UAV="$REPO_ROOT/uav"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

[[ -x "$UAV" ]] || fail "$UAV is not executable"
/usr/bin/bash -n "$UAV"

"$UAV" help >/dev/null
if "$UAV" definitely-not-a-command >/dev/null 2>&1; then
    fail "unknown command unexpectedly returned zero"
fi

if "$UAV" __clean build >/dev/null 2>&1; then
    fail "internal clean command accepted direct entry"
fi

temporary_root="$(mktemp -d /tmp/uav-wrapper-test.XXXXXX)"
trap 'rm -rf -- "$temporary_root"' EXIT
cp "$UAV" "$temporary_root/uav"
chmod +x "$temporary_root/uav"
set +e
bad_output="$(cd "$temporary_root" && ./uav expert-collect --help 2>&1)"
bad_status=$?
set -e
((bad_status != 0)) || fail "incorrect repository assumptions were accepted"
[[ "$bad_output" == *"repository check failed"* ]] || \
    fail "incorrect repository error was unclear"

if /usr/bin/grep -nE 'source[^\n]*uav_ros2_ws' "$UAV"; then
    fail "wrapper contains a legacy workspace source command"
fi
help_output="$("$UAV" help)"
for workflow in build profile expert-collect expert-validate ae-train resnet-train bc-train bc-eval eval bc-eval-plot; do
    [[ "$help_output" == *"$workflow"* ]] || fail "$workflow is missing from help"
done
for workflow in expert-validate ae-train bc-train bc-eval bc-eval-plot; do
    workflow_help="$("$UAV" "$workflow" --help)"
    [[ "$workflow_help" == *"usage:"* ]] || fail "$workflow help is unavailable"
done
resnet_help="$($UAV resnet-train --help)"
[[ "$resnet_help" == *"--dataset DATASET"* ]] || \
    fail "resnet-train help is missing --dataset"
expert_help="$($UAV expert-collect --help)"
[[ "$expert_help" == *"--episodes EPISODES"* ]] || \
    fail "expert-collect help is missing --episodes"
[[ "$expert_help" == *"--resume"* ]] || \
    fail "expert-collect help is missing --resume"
[[ "$expert_help" == *"--seed SEED"* ]] || \
    fail "expert-collect help is missing --seed"
[[ "$expert_help" == *"--scene {v3,v4}"* ]] || \
    fail "expert-collect help is missing --scene"
evaluation_help="$($UAV eval --help)"
[[ "$evaluation_help" == *"--scene {v3,v4}"* ]] || \
    fail "eval alias is missing --scene"
profile_help="$("$UAV" profile current expert-collect --help)"
[[ "$profile_help" == "$expert_help" ]] || fail "profile did not forward collection help"

for retired in \
    offline-check tracking-check mux-check px4-map-check px4-sitl-stream-check bc-overfit \
    status backup doctor test verify ml-test px4-sitl-doctor px4-sitl-flight-check \
    isaac-runtime-flight-check expert-collect-publish expert-runtime-smoke \
    expert-scene-preview artifact-publish topics interfaces shell
do
    set +e
    "$UAV" "$retired" >/dev/null 2>&1
    retired_status=$?
    set -e
    ((retired_status == 64)) || fail "retired command was not rejected: $retired"
done

git -C "$REPO_ROOT" check-ignore -q ros2_ws/build/example
git -C "$REPO_ROOT" check-ignore -q ros2_ws/install/example
git -C "$REPO_ROOT" check-ignore -q ros2_ws/log/example
git -C "$REPO_ROOT" check-ignore -q run_logs/example.log
git -C "$REPO_ROOT" check-ignore -q \
    datasets/bc_expert_highrise_v1/episode_000001/images/frame.jpg

if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$UAV" "$REPO_ROOT/tests/test_uav_wrapper.sh"
fi

printf 'PASS: uav wrapper tests\n'
