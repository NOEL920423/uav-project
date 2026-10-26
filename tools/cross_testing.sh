#!/usr/bin/env bash
set -Eeuo pipefail

trap 'exit 130' INT
trap 'exit 143' TERM

REPO_ROOT="/home/noel_614420090/uav-project"
TRAIN_RUN_ID="run_20261014T184944"
EPISODES=20
SEED=614420090
DEVICE="cuda:0"

RUN_ID="$(TZ=Asia/Taipei date +%Y%m%dT%H%M%S)_$$"
OUTPUT_ROOT="$REPO_ROOT/evaluations/bc/cross_testing/run_${RUN_ID}"
STATUS_FILE="$OUTPUT_ROOT/status.csv"

cd -- "$REPO_ROOT"

# Validate both checkpoints before starting the experiment.
for MODEL_VERSION in v3 v4; do
    CHECKPOINT="$REPO_ROOT/experiments/resnet/cylinder_${MODEL_VERSION}/top/${TRAIN_RUN_ID}/best.pt"

    if [[ ! -f "$CHECKPOINT" ]]; then
        printf 'ERROR: Checkpoint missing: %s\n' "$CHECKPOINT" >&2
        exit 1
    fi
done

# Reserve a new batch directory without overwriting an existing batch.
mkdir -p -- "$REPO_ROOT/evaluations/bc/cross_testing"
mkdir -- "$OUTPUT_ROOT"
mkdir -- "$OUTPUT_ROOT/logs"

printf 'repeat,model,scene,seed,episodes,checkpoint,output,eval_exit_code,log_exit_code\n' \
    > "$STATUS_FILE"

FAILED_GROUPS=0

for REPEAT in 1 2 3; do
    for MODEL_VERSION in v3 v4; do
        CHECKPOINT="$REPO_ROOT/experiments/resnet/cylinder_${MODEL_VERSION}/top/${TRAIN_RUN_ID}/best.pt"

        for SCENE_VERSION in v3 v4; do
            GROUP_ID="repeat_${REPEAT}_model_${MODEL_VERSION}_scene_${SCENE_VERSION}"
            OUTPUT_DIR="$OUTPUT_ROOT/$GROUP_ID"
            LOG_FILE="$OUTPUT_ROOT/logs/${GROUP_ID}.log"

            printf '\nRunning %s\n' "$GROUP_ID"
            printf 'Checkpoint: %s\n' "$CHECKPOINT"
            printf 'Output: %s\n' "$OUTPUT_DIR"

            # The evaluator creates OUTPUT_DIR; do not create it beforehand.
            if ./uav eval \
                --checkpoint "$CHECKPOINT" \
                --image-source top_rgb \
                --scene "$SCENE_VERSION" \
                --episodes "$EPISODES" \
                --seed "$SEED" \
                --device "$DEVICE" \
                --output "$OUTPUT_DIR" \
                2>&1 | tee "$LOG_FILE"
            then
                EVAL_EXIT_CODE=0
                LOG_EXIT_CODE=0
            else
                PIPELINE_CODES=("${PIPESTATUS[@]}")
                EVAL_EXIT_CODE="${PIPELINE_CODES[0]}"
                LOG_EXIT_CODE="${PIPELINE_CODES[1]}"
            fi

            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$REPEAT" "$MODEL_VERSION" "$SCENE_VERSION" \
                "$SEED" "$EPISODES" "$CHECKPOINT" "$OUTPUT_DIR" \
                "$EVAL_EXIT_CODE" "$LOG_EXIT_CODE" \
                >> "$STATUS_FILE"

            if (( EVAL_EXIT_CODE != 0 || LOG_EXIT_CODE != 0 )); then
                FAILED_GROUPS=$((FAILED_GROUPS + 1))
                printf 'ERROR: %s: eval_exit_code=%s log_exit_code=%s\n' \
                    "$GROUP_ID" "$EVAL_EXIT_CODE" "$LOG_EXIT_CODE" >&2
            fi

            # Stop the batch if the evaluation reports an interruption.
            if (( EVAL_EXIT_CODE == 130 || EVAL_EXIT_CODE == 143 )); then
                exit "$EVAL_EXIT_CODE"
            fi
        done
    done
done

printf '\nFinished 12 groups. Groups with command errors: %s\n' \
    "$FAILED_GROUPS"
printf 'Results: %s\n' "$OUTPUT_ROOT"
printf 'Status: %s\n' "$STATUS_FILE"

if (( FAILED_GROUPS > 0 )); then
    exit 1
fi


# cd /home/noel_614420090/uav-project
# bash -n tools/cross_testing.sh && bash tools/cross_testing.sh
