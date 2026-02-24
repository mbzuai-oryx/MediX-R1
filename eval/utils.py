import json
from datasets import load_dataset
from tqdm import tqdm
import requests
import re
from openai import OpenAI
import base64
import io
from filelock import FileLock
import subprocess
import time
import psutil
import os
from PIL import Image


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def load_completed_ids(file_path):
    """Return the set of sample IDs already present in a JSONL file."""
    ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue
    return ids


def load_dataset_with_params(params, dataset_name):
    """Load a dataset from HuggingFace and format samples for evaluation."""
    option_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    os.makedirs("./tmp", exist_ok=True)

    path = params["path"]
    name = params.get("name", "default")
    split = params.get("split", "test")

    if name == "default":
        ds = load_dataset(path, trust_remote_code=True, num_proc=8)[split]
    else:
        ds = load_dataset(path, name=name, trust_remote_code=True, num_proc=8)[split]
    ds_tmp = []
    c = 0

    for sample in tqdm(ds):
        # SLAKE: download images on first run, skip non-English samples
        if dataset_name == "slake_vqa":
            slake_dir = './tmp/slake'
            if not os.path.exists(os.path.join(slake_dir,"imgs.zip")) and not os.path.exists(os.path.join(slake_dir,"imgs")):
                os.system("wget https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip")
                os.system(f"unzip imgs.zip -d {slake_dir}")
            if sample.get("q_lang") != "en":
                continue

        # MIMIC: download images from PhysioNet on first run
        if "mimic" in dataset_name:
            mimic_dir = './tmp/mimic'
            if not os.path.exists(mimic_dir):
                os.makedirs(mimic_dir)
            if os.listdir(mimic_dir) == [] or len(os.listdir(mimic_dir)) < 100:
                image_path = sample.get("image")
                cmd = f"wget -nd -r -N -c -np -P {mimic_dir} --user {os.getenv('PHYSIONET_UNAME','')} --password {os.getenv('PHYSIONET_PWD','')} {image_path}"
                os.system(cmd)
                image_path = image_path.split("/")[-1]
                image_path = os.path.join(mimic_dir, image_path)
                sample["image"] = Image.open(image_path).convert("RGB")
            else:
                image_path = sample.get("image")
                image_path = image_path.split("/")[-1]
                image_path = os.path.join(mimic_dir, image_path)
                sample["image"] = Image.open(image_path).convert("RGB")

        # Build question and answer from dataset config columns
        if isinstance(params.get('question_column'), list):
            question = ""
            for question_column in params.get('question_column'):
                question += f"{question_column.lower().capitalize()}:\n{sample.get(question_column)[0] if isinstance(sample.get(question_column),list) else sample.get(question_column)}\n\n"
            choices = params.get("choices")
            question += f"Choices:"
            for i, option in enumerate(choices):
                if option is not None:
                    question+=f"\n{option_keys[i]}. {option}"
            answer = ""
            for answer_column in params.get("answer"):
                answer += sample.get(answer_column)+"\n"
        else:
            question = f"{sample.get(params.get('question_column'))}" if sample.get(params.get('question_column')) is not None else ""
            choices = []
            if isinstance(params.get("choices_column"), list) and params.get("choices_column") != []:
                question+="\n\nChoices:"
                for choice in params.get("choices_column"):
                    choices.append(sample.get(choice))
            if isinstance(params.get("choices_column"), str):
                question+="\n\nChoices:"
                choices = sample.get(params.get("choices_column"))
                if isinstance(choices, dict):
                    choices = list(choices.values())
            for i, option in enumerate(choices):
                if option is not None:
                    question+=f"\n{option_keys[i]}. {option}"
            if 'answer_id' in params:
                answer = choices[sample.get(params.get("answer_id"))]
            if 'answer_column' in params:
                answer = sample.get(params.get("answer_column"))

        # Resolve image: SLAKE uses img_name, others use image or image_id column
        image_pil = Image.open(os.path.join(slake_dir,"imgs",sample['img_name'])).convert("RGB") if sample.get('img_name') else sample.get("image").convert("RGB") if sample.get("image") else sample.get("image_id").convert("RGB") if sample.get("image_id") else None

        # Override question for MIMIC-specific tasks
        if dataset_name == 'mimic_cxr_report_generation':
            question = """Generate a detailed report based on the scan."""
        if dataset_name == 'mimic_cxr_report_summarization':
            question = f"""Write a short summary of the following report.\n\nReport:\n{sample.get("report")}:"""
            image_pil = None

        sample_updated = {
            "id": f"{dataset_name}_{c}",
            "image": image_pil,
            "question": question.strip(),
            "answer": answer.strip(),
            "answer_idx": sample.get("answer_idx") if "answer_idx" in sample else None,
        }
        ds_tmp.append(sample_updated)
        c+=1
    return ds_tmp


def convert_to_underscored(model_name):
    """Convert 'org/model-name' to 'model_name' for use in file paths."""
    split = model_name.split('/')
    model_name = split[-2] if split[-1].strip() == "" else split[-1]
    return re.sub(r'[^a-zA-Z0-9]', '_', model_name)


def generate(sample, model_name):
    """Generate a model response for a single sample via the vLLM server."""
    image = sample.get("image")
    question = sample.get("question")

    client = OpenAI(
        base_url="http://localhost:8005/v1/",
        api_key="DUMMY_KEY",
    )

    messages = [{"role": "system", "content": [{"type": "text", "text": os.getenv("SYSTEM_PROMPT")}]}] if os.getenv("SYSTEM_PROMPT", None) is not None else []

    # Optionally force chain-of-thought by appending <think> token
    text = question + " <think>" if int(os.getenv("FORCE_THINK", "0")) else question

    if image is not None:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(io.BytesIO(image.save(buf := io.BytesIO(), format='JPEG') or buf.getvalue()).getvalue()).decode()}"
                    }
                },
                {"type": "text", "text": text}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}]
        })

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        timeout=180
    )

    return completion.choices[0].message.content


def _create_eval_client(judge_server='local'):
    """Create an OpenAI-compatible client for the eval judge model.
    Uses OpenRouter when judge_server='openrouter', otherwise uses local vLLM.
    """
    if judge_server == 'openrouter':
        api_key = os.getenv('OPENROUTER_API_KEY')
        assert api_key, "OPENROUTER_API_KEY is not set. Export it before running:\n  export OPENROUTER_API_KEY=your_openrouter_key"
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    else:
        return OpenAI(
            base_url="http://localhost:8005/v1/",
            api_key="DUMMY_KEY",
        )


def _eval_completion(client, model_name, messages):
    """Request a completion from the eval judge model."""
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        timeout=180
    )


def _run_eval_rounds(client, model_name, messages, num_rounds=3):
    """Run multiple evaluation rounds, asking the judge to reevaluate each time.
    Returns list of scores from each round.
    """
    scores = []
    for i in range(num_rounds):
        completion = _eval_completion(client, model_name, messages)
        out = completion.choices[0].message.content
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": out}]
        })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Please reevaluate and make sure your judgment is correct."}]
        })
        if '</think>' in out:
            out = out.split('</think>')[-1].strip()
        elif '</thinking>' in out:
            out = out.split('</thinking>')[-1].strip()
        start = out.find("{")
        end = out.rfind("}") + 1
        out = out[start:end]
        out = json.loads(out)
        scores.append(out['score'])
    return scores, out


def evaluate(sample, model_name="Qwen/Qwen3-14B", judge_server='local'):
    """Score a predicted answer against the ground truth using an LLM judge.
    MIMIC tasks use averaged 0-5 scores; all others use majority-vote binary scoring.
    """
    question = sample["question"]
    correct_answer = sample["answer"]
    predicted_answer = sample["response"]

    if '</think>' in predicted_answer:
        predicted_answer = predicted_answer.split('</think>')[-1].strip()
    elif '</thinking>' in predicted_answer:
        predicted_answer = predicted_answer.split('</thinking>')[-1].strip()

    client = _create_eval_client(judge_server)

    no_think_suffix = " /no_think"

    BASE_PROMPT = f"""You are a medical expert.

Your task is to evaluate whether the Predicted Answer correctly answers the Medical Question, based on the Ground Truth (Correct Answer) provided.

Question:
{question}

Correct Answer:
{correct_answer}

Predicted Answer:
{predicted_answer}

Score 1 if the predicted answer matches the correct answer either fully in text or by indicating the correct option label (e.g., "B", "Option B", or a paraphrased version that clearly identifies the correct choice). Score 0 if the predicted answer is incorrect or points to the wrong option.

Respond strictly in the following JSON format:

```json
{{
"score": <score>
}}
```{no_think_suffix}"""

    MIMIC_PROMPT = f"""You are a medical expert evaluating the clinical accuracy, completeness, and relevance of a generated medical report or summary.

Your task is to compare an AI-generated report or summary to a reference (gold standard) report or summary, based on a clinical instruction or question. Assess the generated output on how well it preserves key clinical information, factual correctness, and clinical reasoning relevant to the task.

Assign a score between 0 and 5 using the following scale:

0 - Completely incorrect: Clinically irrelevant, misleading, or factually wrong. No meaningful alignment with the instruction or reference.

1 - Poor match: Barely relevant or mostly incorrect. Contains significant clinical misinformation or omits nearly all critical details.

2 - Weak match: Some fragments of relevant content are present, but major clinical errors or omissions exist. Clinical utility is low.

3 - Fair match: Contains several relevant points, but includes notable errors, missing findings, or misinterpretations that affect clinical reliability.

4 - Good match: Mostly accurate and clinically sound. Minor issues or missing details, but the overall meaning and purpose are preserved.

5 - Perfect or near-perfect match: Clinically accurate, complete, and faithful to the instruction and reference. No significant omissions or errors.

Respond only in the following example JSON format:

Example JSON format:
```json
{{
"score": <score between 0 and 5>
}}
```

Now, evaluate the following:

### Clinical Instruction or Question::
{question}

### Reference Report or Summary:
{correct_answer}

### AI-Generated Report or Summary:
{predicted_answer}{no_think_suffix}"""

    PROMPT = MIMIC_PROMPT if 'mimic' in sample["id"] else BASE_PROMPT
    messages = [
        {"role": "user", "content": [{"type": "text", "text": PROMPT}]}
    ]

    scores, out = _run_eval_rounds(client, model_name, messages)

    # MIMIC: average the 0-5 scores; others: majority vote on binary 0/1
    if 'mimic' in sample["id"]:
        out['score'] = sum(scores) / len(scores)
    else:
        out['score'] = 1 if scores.count(1) >= 2 else 0

    return out


def process_sample(sample, output_file, model_name):
    """Generate a response for a sample and append to the output JSONL file."""
    response = generate(sample, model_name)
    result = {
        "id": sample["id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "response": response,
        "answer_idx": sample.get("answer_idx"),
    }

    lock_file = f"{output_file}.lock"
    with FileLock(lock_file):
        with open(output_file, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def process_evaluation(sample, eval_file, model_name, judge_server='local'):
    """Evaluate a sample's response and append to the eval JSONL file."""
    evaluation_result = evaluate(sample, model_name, judge_server)

    sample['answer_idx'] = sample.get("answer_idx")
    sample['evaluation'] = evaluation_result
    result = sample

    lock_file = f"{eval_file}.lock"
    with FileLock(lock_file):
        with open(eval_file, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def start_vllm_server(model_name, tensor_parallel_size):
    """Start a vLLM server as a background process and block until it's ready."""
    cmd = [
        "vllm", "serve", model_name,
        "--tensor-parallel-size", tensor_parallel_size,
        "--max-model-len", "512000",
        "--max-num-seqs", "256",
        "--port", "8005",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.95",
        "--enable-chunked-prefill",
    ]

    print(f"Starting VLLM server with model {model_name}...")

    os.makedirs("./logs", exist_ok=True)
    log_file_path = "./logs/vllm_logs.txt"
    print(f"Check {log_file_path} for vLLM logs...")
    log_file = open(log_file_path, "w")

    server_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file
    )

    url = "http://localhost:8005/v1/models"
    print("Waiting for VLLM server to become available...")

    while True:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            print("\nServer is up and model is ready.")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(".", end="", flush=True)
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"\nUnexpected error: {e}")
            break

    return server_process


def kill_server_process(process):
    """Kill the vLLM server process and all its children."""
    if process:
        print("Shutting down VLLM server...")
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()

            gone, still_alive = psutil.wait_procs([parent], timeout=10)
            for p in still_alive:
                p.kill()

            print("VLLM server process terminated successfully")
        except Exception as e:
            print(f"Error killing VLLM server: {e}")
            process.kill()
