#!/bin/bash
# merge_model.sh - Merge sharded FSDP model checkpoints into a single HuggingFace model
#
# This script merges distributed training checkpoint shards (from FSDP/DDP)
# into a single HuggingFace-compatible model that can be used for inference
# or uploaded to the HuggingFace Hub.
#
# Usage:
#   bash merge_model.sh
#
# Before running, update the --local_dir to point to the actor checkpoint
# directory you want to merge. The path should follow the pattern:
#   <checkpoints_root>/<run_name>/global_step_<N>/actor
#
# The merged model will be saved to a "huggingface" subfolder inside --local_dir.
#
# To also upload to HuggingFace Hub, add --hf_upload_path:
#   python3 scripts/model_merger.py --local_dir <path> --hf_upload_path <user/repo>

python3 scripts/model_merger.py --local_dir "path/to/model/actor"
