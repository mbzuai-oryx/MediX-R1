import re

import requests
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1", trust_remote_code=True)

# Reward component weights (must sum to 1.0 when combined with format_weight)
ACCURACY_LLM_WEIGHT = 0.575
ACCURACY_EMBED_WEIGHT = 0.375
MODALITY_WEIGHT = 0.05

# Symbols that indicate a garbage answer
INVALID_SYMBOLS = ["__", ":", "?"]


def format_reward(predict: str) -> float:
    """Checks if the response follows the <thinking>...</thinking><answer>...</answer> format."""
    idx = predict.find("<thinking>")
    if idx == -1:
        return 0.0

    predict_new = predict[idx:].strip()
    pattern = re.compile(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_new)
    return 1.0 if format_match else 0.0


def create_prompt(predict: str, ground_truth: str) -> str:
    return f"""You are a helpful assistant.
Given a predicted answer and the ground truth answer, respond with exactly "YES" if the content matches, otherwise respond with "NO".
Start your response with "YES" or "NO" in uppercase.

Predicted:{predict}
Ground Truth:{ground_truth}
"""


def chat_with_vllm(prompt, model="Qwen/Qwen3-4B-Instruct-2507", url="http://localhost:8085/v1/chat/completions"):
    """Send a prompt to a vLLM server and return the response text."""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_p": 1.0,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def llm_answer_match(predict: str, ground_truth: str) -> float:
    """Use vLLM chat API to ask yes/no if answers match."""
    prompt = create_prompt(predict, ground_truth)
    try:
        reply = chat_with_vllm(prompt)
        return 1.0 if "YES" in reply.strip().upper() else 0.0
    except Exception as e:
        print(f"LLM API error: {e}")
        return 0.0


def _extract_answer(predict: str, ground_truth: str) -> tuple[str, str]:
    """Extract the predicted answer from <answer> tags and clean up ground truth."""
    content_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    given_answer = content_match.group(1).strip() if content_match else predict.strip()
    given_answer = given_answer.strip(".")

    ground_truth = ground_truth.split(">", maxsplit=1)[1].strip()
    ground_truth = ground_truth.strip(".")

    return given_answer, ground_truth


def _is_invalid_answer(answer: str) -> bool:
    """Check if an answer is empty, too short, or contains invalid symbols."""
    if answer == "" or len(answer) == 1:
        return True
    return any(symbol in answer for symbol in INVALID_SYMBOLS)


def accuracy_reward_llm(predict: str, ground_truth: str) -> float:
    """Score accuracy by asking an LLM judge if the answer matches."""
    try:
        given_answer, ground_truth = _extract_answer(predict, ground_truth)

        if _is_invalid_answer(given_answer):
            return 0.0

        if given_answer == ground_truth:
            return 1.0

        return llm_answer_match(given_answer, ground_truth)
    except Exception:
        return 0.0


def accuracy_reward_embed(predict: str, ground_truth: str, threshold: float = 0.8) -> float:
    """Score accuracy using embedding cosine similarity."""
    try:
        given_answer, ground_truth = _extract_answer(predict, ground_truth)

        if _is_invalid_answer(given_answer):
            return 0.0

        if given_answer == ground_truth:
            return 1.0

        embeddings = embed_model.encode([given_answer, ground_truth], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity >= threshold)
    except Exception:
        return 0.0


def modality_reward(predict: str, ground_truth: str) -> float:
    """Reward for correctly predicting the image modality (before the <thinking> tag)."""
    idx = predict.find("<thinking>")
    if idx == -1:
        return 0.0

    predict_modality = predict[:idx].strip()
    ground_truth_modality = ground_truth.split(">", maxsplit=1)[0] + ">"

    return 1.0 if predict_modality.upper() == ground_truth_modality.upper() else 0.0


def compute_score(reward_inputs: list[dict[str, str]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    content_weight = 1 - format_weight
    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input["ground_truth"]

        format_score = format_reward(predict)
        accuracy_score_llm = accuracy_reward_llm(predict, ground_truth)
        accuracy_score_embed = accuracy_reward_embed(predict, ground_truth)
        modality_score = modality_reward(predict, ground_truth)

        overall = (
            content_weight * ACCURACY_LLM_WEIGHT * accuracy_score_llm
            + content_weight * ACCURACY_EMBED_WEIGHT * accuracy_score_embed
            + content_weight * MODALITY_WEIGHT * modality_score
            + format_weight * format_score
        )

        scores.append({
            "overall": overall,
            "format": format_score,
            "accuracy_llm": accuracy_score_llm,
            "accuracy_embed": accuracy_score_embed,
            "modality": modality_score,
        })

    return scores
