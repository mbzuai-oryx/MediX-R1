source .env # load physionet credentials from .env file
export PHYSIONET_UNAME
export PHYSIONET_PWD

# Set environment variables for evaluation
export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200

export OMP_NUM_THREADS=5
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Run evaluation
python eval.py \
    --model MBZUAI/MediX-R1-8B \
    --tasks all \
    --num_workers 128 \
    --generate true \
    --evaluate true \
    --score true \
    --tensor_parallel_size 2