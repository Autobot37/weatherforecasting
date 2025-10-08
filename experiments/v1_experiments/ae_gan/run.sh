#!/usr/bin/env bash
set -euo pipefail

# ==== CONFIG ====
RESUME=${RESUME:-false}  # Can be overridden: RESUME=false ./run.sh
EXPERIMENT_DIR="/home/vatsal/NWM/weatherforecasting/experiments/ae_gan"
OUTPUTS_DIR="${EXPERIMENT_DIR}/outputs/wandb"
TRAIN_SCRIPT="experiments.ae_gan.train"
PYTHON_CMD="python -m"
SUCCESS_MARKER="done"

# ==== FUNCTIONS ====
check_directory_exists() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "⚠️  Directory does not exist: $dir"
        return 1
    fi
    return 0
}

find_latest_checkpoint() {
    local search_dir="$1"
    local latest_ckpt=""
    
    # Check if directory exists first
    if ! check_directory_exists "$search_dir"; then
        echo "📁 Creating outputs directory: $search_dir"
        mkdir -p "$search_dir"
        return 1
    fi
    
    # Search for checkpoints in all wandb subdirectories
    local ckpt_info
    ckpt_info=$(find "$search_dir" -type f -name "epoch=*-step=*.ckpt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1)
    
    if [[ -z "$ckpt_info" ]]; then
        echo "📂 No checkpoints found in: $search_dir"
        return 1
    fi
    
    latest_ckpt=$(echo "$ckpt_info" | cut -d' ' -f2-)
    echo "$latest_ckpt"
    return 0
}

extract_run_info() {
    local ckpt_path="$1"
    local run_dir
    local run_id
    
    # Navigate up to find the run directory (assuming structure: .../run-{timestamp}-{run_id}/...)
    run_dir=$(dirname "$(dirname "$ckpt_path")")
    
    # Extract run_id from directory name (assuming format: run-{timestamp}-{run_id})
    run_id=$(basename "$run_dir" | sed 's/.*-//')
    
    if [[ -z "$run_id" ]]; then
        echo "⚠️  Could not extract run_id from path: $ckpt_path"
        return 1
    fi
    
    echo "$run_id"
    return 0
}

# ==== MAIN LOOP ====
echo "🚀 Starting training loop..."
echo "📁 Experiment directory: $EXPERIMENT_DIR"
echo "📁 Outputs directory: $OUTPUTS_DIR"

while true; do
    if [[ "$RESUME" == "true" ]]; then
        echo "🔄 Attempting to resume from latest checkpoint..."
        
        # Find latest checkpoint
        if latest_ckpt=$(find_latest_checkpoint "$OUTPUTS_DIR"); then
            # Extract run ID
            if run_id=$(extract_run_info "$latest_ckpt"); then
                RESUME_ARG="--resume ${latest_ckpt} --run_id ${run_id}"
                echo "✔️  Resuming from checkpoint: ${latest_ckpt}"
                echo "📋 Run ID: ${run_id}"
            else
                echo "❌ Failed to extract run ID. Starting fresh instead."
                RESUME_ARG=""
                RESUME=false
            fi
        else
            echo "🆕 No checkpoint found. Starting fresh."
            RESUME_ARG=""
            RESUME=false
        fi
    else
        RESUME_ARG=""
        echo "🆕 Starting fresh (RESUME=false)"
    fi
    
    # ==== Launch training ====
    echo "🏃 Launching training..."
    echo "📝 Command: ${PYTHON_CMD} ${TRAIN_SCRIPT} ${RESUME_ARG}"
    
    # Create log file with timestamp
    LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
    
    # Run training and capture exit code
    set +e  # Temporarily disable exit on error
    ${PYTHON_CMD} ${TRAIN_SCRIPT} ${RESUME_ARG} 2>&1 | tee "$LOG_FILE"
    TRAIN_EXIT_CODE=$?
    set -e  # Re-enable exit on error
    
    # ==== Check result ====
    if [[ $TRAIN_EXIT_CODE -eq 0 ]] && grep -q "${SUCCESS_MARKER}" "$LOG_FILE"; then
        echo "✅ Training completed successfully. '${SUCCESS_MARKER}' detected."
        echo "📄 Log file: $LOG_FILE"
        break
    else
        if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
            echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
        else
            echo "❌ Training finished but '${SUCCESS_MARKER}' not found in log."
        fi
        
        echo "🔄 Retrying in 10 seconds..."
        sleep 10
        RESUME=true
    fi
done

echo "🎉 Training completed successfully. Exiting script."
echo "📄 Final log file: $LOG_FILE"