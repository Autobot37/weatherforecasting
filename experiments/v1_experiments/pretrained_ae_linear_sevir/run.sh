
#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="/home/vatsal/NWM/weatherforecasting/experiments/pretrained_ae_linear_sevir/outputs/wandb/latest-run/files/checkpoints"
TRAIN_SCRIPT="experiments.pretrained_ae_linear_sevir.train"
PYTHON_CMD="python -m"
SUCCESS_MARKER="done"    

while true; do
  if compgen -G "${CHECKPOINT_DIR}/epoch=*-step=*.ckpt" > /dev/null; then
    LATEST_CKPT=$(ls -t "${CHECKPOINT_DIR}"/epoch=*-step=*.ckpt | head -n1)
    RESUME_ARG="--resume ${LATEST_CKPT}"
    echo "✔ Resuming from checkpoint: ${LATEST_CKPT}"
  else
    RESUME_ARG=""
    echo "⚠ No checkpoints found; starting from scratch"
  fi

  ${PYTHON_CMD} ${TRAIN_SCRIPT} ${RESUME_ARG} 2>&1 | tee training.log

  if grep -q "${SUCCESS_MARKER}" training.log; then
    echo "✅ ${SUCCESS_MARKER} detected. Exiting loop."
    break
  else
    echo "❌ '${SUCCESS_MARKER}' not found. Retrying in 10s..."
    sleep 10
  fi
done

