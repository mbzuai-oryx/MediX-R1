#!/bin/bash
# run_train.sh - Launch a training run
#
# This is the main entry point for starting training. Change the script path
# below to select which training configuration to run.
#
# Available configs:
#   examples/medix-r1_2b_dapo.sh   
#   examples/medix-r1_7b_grpo.sh   
#   examples/medix-r1_8b_dapo.sh   
#   examples/medix-r1_8b_grpo.sh   
#   examples/medix-r1_30b_dapo.sh 
#
# Usage:
#   bash run_train.sh

bash examples/medix-r1_8b_dapo.sh
