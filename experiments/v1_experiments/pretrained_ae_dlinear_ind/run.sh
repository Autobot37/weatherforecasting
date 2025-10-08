#!/usr/bin/env bash
set -euo pipefail

# ==== CONFIG ====
RESUME=${RESUME:-true}  # Can be overridden: RESUME=false ./run.sh
EXPERIMENT_DIR="/home/vatsal/NWM/weatherforecasting/experiments/pretrained_ae_dlinear_ind"
OUTPUTS_DIR="${EXPERIMENT_DIR}/outputs/wandb"
TRAIN_SCRIPT="experiments.pretrained_ae_dlinear_ind.train"
PYTHON_CMD="python -m"
SUCCESS_MARKER="done"

while true; do
  if [[ "$RESUME" == "true" ]]; then
    # Find latest W&B run directory
    LATEST_RUN_DIR=$(find "$OUTPUTS_DIR" -maxdepth 1 -type d -name "run-*" -printf '%T@ %p\n' | sort -n | tail -n1 | cut -d' ' -f2-)

    if [[ -z "$LATEST_RUN_DIR" ]]; then
      echo "❌ No W&B run found in: $OUTPUTS_DIR"
      exit 1
    fi

    RUN_ID=$(basename "$LATEST_RUN_DIR" | cut -d'-' -f3)
    CKPT_DIR="${LATEST_RUN_DIR}/files/checkpoints"

    if compgen -G "${CKPT_DIR}/epoch=*-step=*.ckpt" > /dev/null; then
      LATEST_CKPT=$(ls -t "${CKPT_DIR}"/epoch=*-step=*.ckpt | head -n1)
      RESUME_ARG="--resume ${LATEST_CKPT} --run_id ${RUN_ID}"
      echo "✔ Resuming from checkpoint: ${LATEST_CKPT}"
    else
      echo "❌ No checkpoint found in ${CKPT_DIR}"
      exit 1
    fi
  else
    RESUME_ARG=""
    echo "🆕 Starting fresh (RESUME=false)"
  fi

  # ==== Launch training ====
  ${PYTHON_CMD} ${TRAIN_SCRIPT} ${RESUME_ARG} 2>&1 | tee training.log

  # ==== Check result ====
  if grep -q "${SUCCESS_MARKER}" training.log; then
    echo "✅ '${SUCCESS_MARKER}' detected. Exiting loop."
    break
  else
    echo "❌ '${SUCCESS_MARKER}' not found. Retrying in 10s..."
    sleep 10
    RESUME=true
  fi
done
