#!/usr/bin/env bash
set -e

PYTHON_SCRIPT="/home/vatsal/NWM/weatherforecasting/experiments/pretrained_ae_conv_disc/train.py"
SUCCESS_MARKER="done"
RESUME_FLAG="--resume"
RESUME=true

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
        
        echo -e "${CYAN}Command: python $PYTHON_SCRIPT $config_args $resume_param${NC}"
        
        if python "$PYTHON_SCRIPT" $config_args $resume_param 2>&1 | tee /tmp/training_output.log; then
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

declare -a RUNS=(
    # Config 1: Conservative setup - low peak lr, late disc start, low disc weight
    "experiment_name=pae_6e6_04_05_09_w1 cosine_warmup.peak_lr=5e-6 cosine_warmup.start_lr=5e-7 cosine_warmup.final_lr=5e-8 lpips.disc_start=0.4 lpips.disc_peak_lr=5e-6 lpips.disc_start_lr=5e-7 lpips.disc_final_lr=5e-8 lpips.disc_beta1=0.5 lpips.disc_weight=1.0 optim.lr=5e-6 optim.beta1=0.9"
    
    # Config 3: Balanced mid-range - medium peak lr, medium disc start, high disc weight
    "experiment_name=pae_5e5_02_09_09_w2 cosine_warmup.peak_lr=5e-5 cosine_warmup.start_lr=5e-6 cosine_warmup.final_lr=5e-8 lpips.disc_start=0.2 lpips.disc_peak_lr=5e-5 lpips.disc_start_lr=5e-6 lpips.disc_final_lr=5e-8 lpips.disc_beta1=0.9 lpips.disc_weight=2.0 optim.lr=5e-5 optim.beta1=0.9"
    
    # Config 4: High lr with conservative disc and low disc weight
    "experiment_name=pae_4e4_04_09_05_w1 cosine_warmup.peak_lr=5e-4 cosine_warmup.start_lr=5e-5 cosine_warmup.final_lr=5e-7 lpips.disc_start=0.4 lpips.disc_peak_lr=5e-5 lpips.disc_start_lr=5e-6 lpips.disc_final_lr=5e-8 lpips.disc_beta1=0.9 lpips.disc_weight=1.0 optim.lr=5e-4 optim.beta1=0.5"
    
    # # Config 5: Low lr with aggressive disc and high disc weight
    "experiment_name=pae_6e6_01_05_05_w2 cosine_warmup.peak_lr=5e-6 cosine_warmup.start_lr=5e-7 cosine_warmup.final_lr=5e-8 lpips.disc_start=0.1 lpips.disc_peak_lr=5e-4 lpips.disc_start_lr=5e-5 lpips.disc_final_lr=5e-7 lpips.disc_beta1=0.5 lpips.disc_weight=2.0 optim.lr=5e-6 optim.beta1=0.5"
    
    # Config 6: Medium lr, early disc start, low disc weight
    "experiment_name=pae_5e5_01_09_05_w1 cosine_warmup.peak_lr=5e-5 cosine_warmup.start_lr=5e-6 cosine_warmup.final_lr=5e-8 lpips.disc_start=0.1 lpips.disc_peak_lr=5e-4 lpips.disc_start_lr=5e-5 lpips.disc_final_lr=5e-7 lpips.disc_beta1=0.9 lpips.disc_weight=1.0 optim.lr=5e-5 optim.beta1=0.5"
    
    # Config 7: High lr, late disc start, high disc weight
    "experiment_name=pae_4e4_04_05_09_w2 cosine_warmup.peak_lr=5e-4 cosine_warmup.start_lr=5e-5 cosine_warmup.final_lr=5e-7 lpips.disc_start=0.4 lpips.disc_peak_lr=5e-6 lpips.disc_start_lr=5e-7 lpips.disc_final_lr=5e-8 lpips.disc_beta1=0.5 lpips.disc_weight=2.0 optim.lr=5e-4 optim.beta1=0.9"
    
    # Config 8: High lr, medium disc start, low disc weight
    "experiment_name=pae_4e4_02_09_05_w1 cosine_warmup.peak_lr=5e-4 cosine_warmup.start_lr=5e-5 cosine_warmup.final_lr=5e-7 lpips.disc_start=0.2 lpips.disc_peak_lr=5e-4 lpips.disc_start_lr=5e-5 lpips.disc_final_lr=5e-7 lpips.disc_beta1=0.9 lpips.disc_weight=1.0 optim.lr=5e-4 optim.beta1=0.5"
)

for config in "${RUNS[@]}"; do
    run_with_retry "$config"
done

echo -e "${GREEN}🎉 All experiments completed!${NC}"