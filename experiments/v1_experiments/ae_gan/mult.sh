#!/usr/bin/env bash
set -e
CONFIG=config.yaml
PYTHON_SCRIPT=train.py  # replace with actual script
PROJECT_NAME="ae_gan_night"

declare -a RUNS=(
  ""
)

for OVERRIDE in "${RUNS[@]}"; do
  echo "Running with: $OVERRIDE"
  python "$PYTHON_SCRIPT" $OVERRIDE
done