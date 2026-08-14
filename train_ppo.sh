#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./train_ppo.sh [quick|50k|100k|TIMESTEPS] [extra PPO options]
  ./train_ppo.sh resume RUN_DIRECTORY [50k|100k|TIMESTEPS] [extra PPO options]

Examples:
  ./train_ppo.sh 50k
  ./train_ppo.sh 100k
  ./train_ppo.sh 25000 --evaluation-episodes 30
  ./train_ppo.sh quick --no-progress
  ./train_ppo.sh resume training_runs/ppo_city_50k_20260811_190000 50k

The script automatically selects the BC checkpoint, Isaac headless camera mode,
standard PPO hyperparameters, and a timestamped output directory.
EOF
}

resume_checkpoint=""
resume_output=""
if [[ "${1:-}" == "resume" ]]; then
    [[ -n "${2:-}" ]] || { printf 'ERROR: resume requires a run directory\n' >&2; exit 2; }
    resume_output="${2%/}"
    resume_checkpoint="$resume_output/latest.pt"
    [[ -f "$REPO_ROOT/$resume_checkpoint" || -f "$resume_checkpoint" ]] || {
        printf 'ERROR: resume checkpoint is missing: %s\n' "$resume_checkpoint" >&2
        exit 2
    }
    set -- "${3:-50k}" "${@:4}"
fi

case "${1:-50k}" in
    help|-h|--help)
        usage
        exit 0
        ;;
    quick)
        timesteps=8192
        evaluation_episodes=20
        label=quick
        ;;
    50k)
        timesteps=50000
        evaluation_episodes=50
        label=50k
        ;;
    100k)
        timesteps=100000
        evaluation_episodes=100
        label=100k
        ;;
    *[!0-9]*|'')
        printf 'ERROR: expected quick, 50k, 100k, or a positive timestep count\n' >&2
        usage >&2
        exit 2
        ;;
    *)
        timesteps="$1"
        ((timesteps > 0)) || { printf 'ERROR: TIMESTEPS must be positive\n' >&2; exit 2; }
        evaluation_episodes=50
        label="${timesteps}steps"
        ;;
esac
shift || true

if [[ -n "$resume_output" ]]; then
    output="$resume_output"
else
    run_stamp="$(date +%Y%m%d_%H%M%S)"
    output="training_runs/ppo_city_${label}_${run_stamp}"
fi

cd "$REPO_ROOT"
printf 'PPO training preset\n'
printf '  timesteps:          %s\n' "$timesteps"
printf '  evaluation episodes: %s\n' "$evaluation_episodes"
printf '  BC checkpoint:       training_runs/latent_bc_city_v0/best.pt\n'
printf '  output:              %s/%s\n\n' "$REPO_ROOT" "$output"

resume_arguments=()
if [[ -n "$resume_checkpoint" ]]; then
    resume_arguments=(--resume "$resume_checkpoint")
fi

./uav ppo-train \
    --headless \
    --enable_cameras \
    --bc-checkpoint training_runs/latent_bc_city_v0/best.pt \
    --output "$output" \
    --timesteps "$timesteps" \
    --rollout-steps 512 \
    --update-epochs 5 \
    --minibatch-size 128 \
    --learning-rate 3e-5 \
    --evaluation-episodes "$evaluation_episodes" \
    --seed 614420090 \
    "${resume_arguments[@]}" \
    "$@"

summary_path="$output/summary.json"
plot_path="$output/training_metrics.png"
if [[ -f "$summary_path" ]]; then
    python3 -m uav_ml.tools.plot_city_training_results \
        --ppo-run "$output" \
        --output "$plot_path"
    plot_absolute="$(realpath "$plot_path")"
    summary_absolute="$(realpath "$summary_path")"
    printf '\nTraining result files\n'
    printf '  chart:  %s\n' "$plot_absolute"
    printf '  metrics: %s\n' "$summary_absolute"
else
    printf '\nTraining did not reach final evaluation; no result chart was generated.\n'
    printf 'Resume from the latest.pt path printed above.\n'
fi
