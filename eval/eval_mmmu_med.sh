#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com  # uncomment if you need a HuggingFace mirror

EVAL_DATASETS=MMMU-Medical-val
DATASETS_PATH="hf"

MODEL_NAME="Qwen3-VL"
MODEL_PATH="/data0/EVAL_2/RL_MODELS/qwen3_vl_8b_medical_global_step_230"
OUTPUT_NAME="qwen3_vl_8b_medical_global_step_230-MMMU-Medical-val"

OUTPUT_PATH=results/${OUTPUT_NAME}

#vllm setting
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE="2"
USE_VLLM="True"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# Not required for MMMU, but need to keep this judge variable for MedEvalKt to run
JUDGE_MODEL_TYPE="openai"  # openai or gemini or deepseek or claude

# pass hyperparameters and run python sccript
python MedEvalKit/eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES"
