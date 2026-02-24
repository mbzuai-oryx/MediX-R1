#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct

echo "Running GRPO training with model path: $MODEL_PATH"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=MBZUAI/medix-rl-data@train \
    data.val_files=MBZUAI/medix-rl-data@test \
    data.max_prompt_length=4352 \
    data.max_response_length=4096 \
    data.format_prompt=./examples/format_prompt/medical_format.jinja \
    data.answer_key=solution \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/medical.py:compute_score \
    worker.rollout.max_num_batched_tokens=8448 \
    trainer.experiment_name=medix-r1_8b_grpo \
    trainer.total_epochs=2 \
    trainer.val_generations_to_log=15 \
    trainer.save_checkpoint_path=./checkpoints/medix-r1_8b_grpo \
    trainer.n_gpus_per_node=8
