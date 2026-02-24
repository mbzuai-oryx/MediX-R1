#!/bin/bash
# submit.sh - Zip all result files for the evaluated model
#
# Usage:
#   bash submit.sh                              # uses MODEL from eval.sh
#   bash submit.sh "MBZUAI/MediX-R1-8B"        # specify model explicitly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Model name is required as an argument
if [ -z "$1" ]; then
    echo "Usage: bash submit.sh <model_name>"
    echo ""
    echo "Example:"
    echo "  bash submit.sh \"MBZUAI/MediX-R1-8B\""
    exit 1
fi

MODEL="$1"

# Convert model name to underscored format (same logic as utils.py convert_to_underscored)
# Strip org prefix (everything before last '/'), then replace non-alphanumeric with '_'
MODEL_UNDERSCORED=$(echo "$MODEL" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9]/_/g')

MODEL_DIR="${RESULTS_DIR}/${MODEL_UNDERSCORED}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Results directory not found: ${MODEL_DIR}"
    echo "Make sure eval.sh has completed successfully for model: ${MODEL}"
    exit 1
fi

# Also check for MMMU-Medical results (uses original model short name, not underscored)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MMMU_DIR_NAME="${MODEL_SHORT}-MMMU-Medical-val"
MMMU_DIR="${RESULTS_DIR}/${MMMU_DIR_NAME}"

# Generate a unique submission ID from the zip file name as seed
BASE_ZIP_NAME="${MODEL_UNDERSCORED}_submission"
SUBMISSION_ID=$(echo -n "$BASE_ZIP_NAME" | md5sum | cut -c1-8)
ZIP_NAME="${SUBMISSION_ID}_${BASE_ZIP_NAME}.zip"
ZIP_PATH="${RESULTS_DIR}/${ZIP_NAME}"

cd "$RESULTS_DIR"
zip -r "$ZIP_PATH" "${MODEL_UNDERSCORED}/"

if [ -d "$MMMU_DIR" ]; then
    echo "Including MMMU-Medical results: ${MMMU_DIR_NAME}/"
    zip -r "$ZIP_PATH" "${MMMU_DIR_NAME}/"
fi

echo ""
echo "================================================"
echo "Submission zip created:"
echo "  ${ZIP_PATH}"
echo "================================================"
echo ""
echo "Submit your results for verification and"
echo "inclusion on the leaderboard:"
echo ""
echo "  Upload the zip:"
echo "    ${ZIP_PATH}"
echo "  via the submission form:"
echo "    https://forms.gle/H9JaYyyQQFarMq6KA"
echo ""
echo "  Or submit from the leaderboard page:"
echo "    https://medix.cvmbzuai.com/leaderboard"
echo ""
echo "  Project page: https://medix.cvmbzuai.com"
echo "================================================"
