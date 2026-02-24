# MediX-R1: Training Code

Training code for the MediX-R1 series of medical VLMs using composite reinforcement rewards.

## Models

| Config | Algorithm | Script |
|--------|-----------|--------|
| MediX-R1 2B | DAPO | `examples/medix-r1_2b_dapo.sh` |
| MediX-R1 7B | GRPO | `examples/medix-r1_7b_grpo.sh` |
| MediX-R1 8B | DAPO | `examples/medix-r1_8b_dapo.sh` |
| MediX-R1 8B | GRPO | `examples/medix-r1_8b_grpo.sh` |
| MediX-R1 30B | DAPO | `examples/medix-r1_30b_dapo.sh` |

## Dataset

Training and evaluation use the [MBZUAI/medix-rl-data](https://huggingface.co/datasets/MBZUAI/medix-rl-data) dataset, which contains medical VQA samples with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `image` | Sequence[Image] | Medical image(s) for the question |
| `problem` | string | The medical question |
| `solution` | string | Ground truth answer |

The dataset has two splits:
- **train**: 51.3k samples
- **test**: 2.45k samples


## Installation

### Option 1: Docker

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

If your environment does not support Docker, you can use **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Option 2: Pip Install

```bash
git clone https://github.com/mbzuai-oryx/MediX-R1.git
cd MediX-R1/training
pip install -e .
```

### Software Requirements

- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3


## Training

### Step 1: Start the vLLM reward server

The LLM-based reward function calls a vLLM server to judge answer correctness. Start it **before** training:

```bash
bash vllm_serve.sh
```

Edit `vllm_serve.sh` to configure the model, GPU IDs, port, and other settings. The default serves `Qwen/Qwen3-4B-Instruct-2507` on port 8085.

Make sure the model name and port in `vllm_serve.sh` match those in `examples/reward_function/medical.py`'s `chat_with_vllm()` function.

### Step 2: Start training

Edit `run_train.sh` to select which training config to run, then:

```bash
bash run_train.sh
```

Or run a specific config directly:

```bash
bash examples/medix-r1_8b_dapo.sh
```

### Step 3: Merge checkpoint

After training, merge the FSDP sharded checkpoint into a HuggingFace-compatible model:

```bash
bash merge_model.sh
```

Edit `merge_model.sh` to point `--local_dir` to your checkpoint's actor directory.

## Acknowledgements

Built on top of [EasyR1](https://github.com/hiyouga/EasyR1), which is a fork of [veRL](https://github.com/volcengine/verl).
