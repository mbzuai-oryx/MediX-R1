source .env # load physionet credentials from .env file
export PHYSIONET_UNAME
export PHYSIONET_PWD

# Set environment variables for evaluation
export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200

export OMP_NUM_THREADS=5
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Set this (OPENROUTER_API_KEY) if you want to use Qwen3-14B as Judge from OpenRouter for evaluation, 
# otherwise it will default to local judge with Qwen3-14B using vLLM. 
# Note that using local judge will require more resources and time for evaluation.

export OPENROUTER_API_KEY=""

# Model to evaluate (edit this)
MODEL="MBZUAI/MediX-R1-8B"

# Run evaluation
python eval.py \
    --model "$MODEL" \
    --tasks mmlu_anatomy \
    --num_workers 128 \
    --generate true \
    --evaluate true \
    --score true \
    --tensor_parallel_size 2 \
    --judge_server local
# set --judge_server to "openrouter" or "local" (see above)

if [ $? -ne 0 ]; then
    echo ""
    echo "Evaluation failed. Check the error above."
    exit 1
fi

MODEL_UNDERSCORED=$(echo "$MODEL" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9]/_/g')
BASE_ZIP_NAME="${MODEL_UNDERSCORED}_submission"
SUBMISSION_ID=$(echo -n "$BASE_ZIP_NAME" | md5sum | cut -c1-8)
ZIP_PATH="results/${SUBMISSION_ID}_${BASE_ZIP_NAME}.zip"

echo ""
echo "================================================"
echo "Evaluation complete!"
echo ""
echo "To submit your results for verification and"
echo "inclusion on the leaderboard:"
echo ""
echo "  1. Run: bash submit.sh <model_name> (same as the one used in eval.sh)"
echo "  2. Upload the generated zip:"
echo "     ${ZIP_PATH}"
echo "     via the submission form:"
echo "     https://forms.gle/H9JaYyyQQFarMq6KA"
echo ""
echo "You can also submit from the leaderboard page:"
echo "  https://medix.cvmbzuai.com/leaderboard"
echo ""
echo "Project page: https://medix.cvmbzuai.com"
echo "================================================"