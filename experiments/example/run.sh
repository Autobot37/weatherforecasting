#!/usr/bin/env bash
set -e

PYTHON_SCRIPT="/home/vatsal/NWM/weatherforecasting/experiments/ae_gan_kl/train.py" # Path to your training script
SUCCESS_MARKER="done" # done is printed at the end of successful training in train.py so we can use it to check if script finished successfully
RESUME_FLAG="--resume" # to indicate whether to resume training from latest checkpoint or not
RESUME=true # default value for resume flag is true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' 

# runs a training configuration till it didn't detect SUCCESS_MARKER in the output log
run_with_retry() {
    local config_args="$1"
    local current_resume="$RESUME"
    
    echo -e "${PURPLE}=== Running: $config_args ===${NC}"
    
    while true; do
        local resume_param=""
        if [ "$current_resume" = true ]; then
            resume_param="$RESUME_FLAG True"
        else
            resume_param="$RESUME_FLAG False"
        fi
        
        echo -e "${CYAN}Command: python $PYTHON_SCRIPT $config_args $resume_param${NC}"
        
        if python "$PYTHON_SCRIPT" $config_args $resume_param 2>&1 | tee /tmp/training_output.log; then
            if grep -q "$SUCCESS_MARKER" /tmp/training_output.log; then
                echo -e "${GREEN}✓ Training completed successfully!${NC}"
                break #break only if training was successful and SUCCESS_MARKER was found
            fi
        else
            echo -e "${RED}⚠ Training failed. Retrying with resume...${NC}"
            current_resume=true
        fi
        
        sleep 5
    done
}

# Define your different runs here. config.yaml is overridden by these command line args
declare -a RUNS=(
    "experiment_name=ae_gan_kl optim.lr=1e-4",
    "experiment_name=ae_gan_kl_disc_1 lpips.disc_start=0.0 lpips.disc_weight=1.0",
)

for config in "${RUNS[@]}"; do
    run_with_retry "$config"
done

echo -e "${GREEN}🎉 All experiments completed!${NC}"