# MediX-R1 Evaluation

Evaluation pipeline for medical LLM and VLM models. Supports response generation via vLLM, LLM-as-judge evaluation, and score aggregation.

## Setup

### 1. Create a conda environment

```bash
conda create -n medixr1_eval python=3.11 -y
conda activate medixr1_eval
```

### 2. Install uv and dependencies

```bash
pip install uv
uv pip install -r requirements.txt
```

### 3. Configure credentials (MIMIC-CXR only)

If you plan to evaluate on MIMIC-CXR tasks (`mimic_cxr_report_generation`, `mimic_cxr_report_summarization`), create a `.env` file in the `eval/` directory with your [PhysioNet](https://physionet.org/) credentials:

```
PHYSIONET_UNAME=your_username
PHYSIONET_PWD=your_password
```

This is not required for any other tasks and can be skipped.

## Running the Evaluation

The pipeline has three phases that can be run independently or together: **generate**, **evaluate**, and **score**.

### Quick start (all phases at once)

```bash
bash eval.sh
```

### Running phases individually

> **Note:** The examples below use `--num_workers 128`. Reduce this based on your machine's available CPU and memory (e.g. `--num_workers 8` for smaller machines).

#### Step 1: Generate responses

Starts a vLLM server with the candidate model, generates responses for each sample, and saves them to a JSONL file. Supports resuming -- already processed sample IDs are skipped automatically.

```bash
python eval.py \
    --model MBZUAI/MediX-R1-8B \
    --tasks all \
    --num_workers 128 \
    --generate true \
    --tensor_parallel_size 2
```

#### Step 2: Evaluate responses

Uses an LLM judge to score each response against the ground truth. Starts a vLLM server for the judge model. Also supports resuming.

```bash
python eval.py \
    --model MBZUAI/MediX-R1-8B \
    --eval_model Qwen/Qwen3-14B \
    --tasks all \
    --num_workers 128 \
    --evaluate true \
    --tensor_parallel_size 2
```

**Alternative: Use OpenRouter instead of self-hosting the judge model**

If you don't have enough GPU resources to run the judge model locally, you can use [Qwen3-14B via OpenRouter](https://openrouter.ai/qwen/qwen3-14b). Export your OpenRouter credentials before running:

```bash
export OPENROUTER_API_KEY=your_openrouter_key

python eval.py \
    --model MBZUAI/MediX-R1-8B \
    --eval_model qwen/qwen3-14b \
    --judge_server openrouter \
    --tasks all \
    --num_workers 128 \
    --evaluate true
```

#### Step 3: Aggregate scores

Computes per-task accuracy and prints a summary table. No GPU required.

```bash
python eval.py \
    --model MBZUAI/MediX-R1-8B \
    --tasks all \
    --score true
```

## Task Selection

Tasks are defined in `tasks.yaml` under two categories: `llm` and `vlm`.

| Selector | Description |
|---|---|
| `all` | Run all tasks |
| `llm` | Run all text-only tasks |
| `vlm` | Run all vision-language tasks |
| `<task_name>` | Run a specific task (e.g. `medqa`, `slake_vqa`) |

Multiple tasks can be specified: `--tasks medqa slake_vqa pmc_vqa`

### Available tasks

**LLM** (text-only): `mmlu_clinical_knowledge`, `mmlu_college_biology`, `mmlu_college_medicine`, `mmlu_medical_genetics`, `mmlu_professional_medicine`, `mmlu_anatomy`, `medmcqa`, `medqa`, `usmle_sa`, `pubmedqa`, `mimic_cxr_report_summarization`

**VLM** (vision-language): `slake_vqa`, `rad_vqa`, `path_vqa`, `pmc_vqa`, `pmc_vqa_hard`, `mimic_cxr_report_generation`

## Output Structure

```
results/
  <model_name>/
    <model_name>.jsonl            # Generated responses
    <model_name>_eval.jsonl       # Evaluation results
    <model_name>_score.txt        # Final score table
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--tasks` | *(required)* | Tasks to run (see task selection above) |
| `--model` | `MBZUAI/MediX-R1-8B` | Candidate model for generation |
| `--eval_model` | `Qwen/Qwen3-14B` | Judge model for evaluation (supports OpenRouter as alternative) |
| `--config` | `tasks.yaml` | Path to dataset config file |
| `--output_dir` | `results` | Output directory |
| `--num_workers` | `1` | Parallel workers for generation/evaluation |
| `--tensor_parallel_size` | `1` | vLLM tensor parallelism (number of GPUs) |
| `--judge_server` | `local` | Judge server: `local` (vLLM) or `openrouter` |
| `--generate` | `false` | Run response generation phase |
| `--evaluate` | `false` | Run LLM judge evaluation phase |
| `--score` | `false` | Run score aggregation phase |

---

## Evaluating on MMMU-Medical

MMMU-Medical evaluation uses the [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit) framework and is run separately from the main pipeline above.

> **Note:** MedEvalKit has its own dependency versions that may conflict with the `medixr1_eval` environment. It is recommended to create a separate conda environment for it.

### Step 1: Clone MedEvalKit

```bash
cd eval/
git clone https://github.com/alibaba-damo-academy/MedEvalKit.git
```

### Step 2: Create a separate environment and install dependencies

```bash
conda create -n medevalkit python=3.11 -y
conda activate medevalkit
pip install uv
cd MedEvalKit
uv pip install -r requirements.txt
cd ..
```

### Step 3: Update vllm

Update vllm to the latest version to ensure support for newer models:

```bash
uv pip install -U vllm
```

### Step 4: Copy the Qwen3-VL model config

MedEvalKit needs a model adapter to run Qwen3-VL. Copy the provided one into the cloned repo:

```bash
cp -r MedEvalKit_ModelFile/Qwen3_VL/ MedEvalKit/models/
```

### Step 5: Register Qwen3-VL in MedEvalKit

Add the following code to `MedEvalKit/LLM.py` to register Qwen3-VL as an available model:

```python
@LLMRegistry.register("Qwen3-VL")
class Qwen3_VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Qwen3_VL.Qwen3_VL_vllm import Qwen3_VL
        return Qwen3_VL(model_path, args)
```

### Step 6: Run the evaluation

Edit `eval_mmmu_med.sh` to set `MODEL_PATH` to your model checkpoint, then run:

```bash
bash eval_mmmu_med.sh
```

### MedEvalKit Output Structure

Results are saved under the MedEvalKit `eval_results/` directory, organized by model and subject:

```
results/
  <model_name>/
    total_results.json                          # Overall scores
    MMMU-Medical-val/
      result.json                               # Aggregate result across subjects
      Basic_Medical_Science/
        output_sample.json                      # Raw model outputs
        parsed_output.json                      # Parsed answers
        result.json                             # Subject-level scores
      Clinical_Medicine/
        ...
      Diagnostics_and_Laboratory_Medicine/
        ...
      Pharmacy/
        ...
      Public_Health/
        ...
```

---

## Submitting Results

After evaluation completes, use `submit.sh` to package your results for submission to the leaderboard.

### Step 1: Generate the submission zip

```bash
bash submit.sh
```

This zips the entire `results/<model_name>/` directory. If MMMU-Medical results exist (`results/<model_name>-MMMU-Medical-val/`), they are included automatically.

### Step 2: Submit the zip

Upload the generated zip via the submission form:

> https://forms.gle/H9JaYyyQQFarMq6KA

You can also submit from the leaderboard page:

> https://medix.cvmbzuai.com/leaderboard

Project page: https://medix.cvmbzuai.com

### Submission zip structure

When extracted, the zip contains:

```
MediX_R1_8B/
├── MediX_R1_8B.jsonl                  # Generated responses
├── MediX_R1_8B_eval.jsonl             # Evaluation results
└── MediX_R1_8B_score.txt              # Final score table
MediX-R1-8B-MMMU-Medical-val/          # (included if present)
└── MMMU-Medical-val/
    ├── result.json
    ├── Basic_Medical_Science/
    │   ├── output_sample.json
    │   ├── parsed_output.json
    │   └── result.json
    ├── Clinical_Medicine/
    │   └── ...
    ├── Diagnostics_and_Laboratory_Medicine/
    │   └── ...
    ├── Pharmacy/
    │   └── ...
    └── Public_Health/
        └── ...
```
