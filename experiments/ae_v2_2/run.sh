#!/usr/bin/env bash
set -e

PYTHON_SCRIPT="/home/vatsal/NWM/weatherforecasting/experiments/ae_v2_2/train_data2.py"
SUCCESS_MARKER="done"
RESUME_FLAG="--resume"
RESUME=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' 

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
        
        echo -e "${CYAN}Command: $TASKSET_CMD python $PYTHON_SCRIPT $config_args $resume_param${NC}"

        if $TASKSET_CMD python "$PYTHON_SCRIPT" $config_args $resume_param 2>&1 | tee /tmp/training_output.log; then
            if grep -q "$SUCCESS_MARKER" /tmp/training_output.log; then
                echo -e "${GREEN}✓ Training completed successfully!${NC}"
                break
            else
                echo -e "${YELLOW}⚠ Training finished but no success marker found. Retrying with resume...${NC}"
                current_resume=true
            fi
        else
            echo -e "${RED}⚠ Training failed. Retrying with resume...${NC}"
            current_resume=true
        fi
        
        sleep 5
    done
}

TASKSET_CMD="taskset -c 0-8"

declare -a RUNS=(
    project_name=ae_test_v2
)

for config in "${RUNS[@]}"; do
    run_with_retry "$config"
done

echo -e "${GREEN}🎉 All experiments completed!${NC}"