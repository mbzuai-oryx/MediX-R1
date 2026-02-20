import argparse
from tqdm import tqdm
import yaml
import os
import json
import concurrent.futures
import pandas as pd
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on medical datasets')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks/datasets to evaluate on')
    parser.add_argument('--config', type=str, default='tasks.yaml', help='Path to the dataset configuration YAML file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for parallel processing')
    parser.add_argument('--generate', type=bool, default=False, help='Generate responses for the given tasks')
    parser.add_argument('--evaluate', type=bool, default=False, help='Evaluate the generated responses')
    parser.add_argument('--score', type=bool, default=False, help='Calculate scores for the evaluation results')
    parser.add_argument('--model', type=str, default='MBZUAI/MediX-R1-8B', help='Model name for generation')
    parser.add_argument('--eval_model', type=str, default='Qwen/Qwen3-14B', help='Model name for evaluation')
    parser.add_argument('--tensor_parallel_size', type=str, default=1, help='Tensor parallel size for vllm server')
    args = parser.parse_args()

    # Load dataset configurations from YAML file
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as file:
                dataset_configs = yaml.safe_load(file)
                print(f"Loaded dataset configurations from {args.config}")
        else:
            print(f"Configuration file {args.config} not found.")
            return
    except Exception as e:
        print(f"Error loading configuration from {args.config}: {e}")
        return

    # Build flat task list and config lookup from the nested YAML structure
    datasets_to_load = []
    configs_to_load = {}
    all_tasks = []
    for cat in dataset_configs.keys():
        all_tasks += list(dataset_configs[cat].keys())
        for task in dataset_configs[cat]:
            configs_to_load[task] = dataset_configs[cat][task]

    for task in args.tasks:
        if task in all_tasks:
            datasets_to_load.append(task)
        elif task == "all":
            datasets_to_load += all_tasks
            break
        elif task in dataset_configs:
            datasets_to_load += list(dataset_configs[task].keys())
        else:
            print(f"Warning: Unknown dataset '{task}'")

    if not datasets_to_load:
        print("No valid datasets selected. Available options:")
        print(f"- Individual datasets: {', '.join(all_tasks)}")
        print(f"- Groups: all, {', '.join(dataset_configs.keys())}")
        return

    # Set up output paths
    model_name = args.model
    model_name_underscored = convert_to_underscored(model_name)
    os.makedirs(os.path.join(args.output_dir, model_name_underscored), exist_ok=True)
    output_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}.jsonl")
    eval_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_eval.jsonl")

    print(f"Model: {model_name}")
    print(f"Selected datasets: {datasets_to_load}")
    print(f"Results will be saved to: {output_file}")

    candidate_server = False
    eval_server = False

    # --- Phase 1: Generate responses ---
    if args.generate:
        try:
            # Load all datasets once to determine total expected count
            all_samples = []
            for dataset_name in datasets_to_load:
                print(f"Loading dataset: {dataset_name}")
                all_samples += load_dataset_with_params(configs_to_load[dataset_name], dataset_name)
            total_expected = len(all_samples)

            while True:
                completed_ids = load_completed_ids(output_file)
                remaining = [s for s in all_samples if s["id"] not in completed_ids]

                if not remaining:
                    print(f"All {total_expected} samples generated.")
                    break

                print(f"Generating responses for {len(remaining)}/{total_expected} samples using {args.num_workers} workers...")

                if not candidate_server:
                    candidate_server = start_vllm_server(model_name, args.tensor_parallel_size)

                if args.num_workers <= 1:
                    for sample in tqdm(remaining):
                        process_sample(sample, output_file, model_name)
                else:
                    from functools import partial
                    worker_func = partial(process_sample, output_file=output_file, model_name=model_name)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                        futures = [executor.submit(worker_func, sample) for sample in remaining]
                        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            pass

                # Re-check completeness after this pass
                completed_count = len(load_completed_ids(output_file))
                if completed_count >= total_expected:
                    print(f"All {total_expected} samples generated.")
                    break
                else:
                    print(f"Generated {completed_count}/{total_expected} samples. Retrying missing ones...")

        finally:
            if candidate_server:
                kill_server_process(candidate_server)

    # --- Phase 2: Evaluate responses with LLM judge ---
    if args.evaluate:
        try:
            eval_model = args.eval_model
            print(f"Evaluating results using model: {eval_model}")
            all_responses = load_jsonl(output_file)
            total_expected = len(all_responses)

            while True:
                evaluated_ids = load_completed_ids(eval_file)
                remaining = [s for s in all_responses if s["id"] not in evaluated_ids]

                if not remaining:
                    print(f"All {total_expected} responses evaluated.")
                    break

                print(f"Evaluating {len(remaining)}/{total_expected} responses using {args.num_workers} workers...")

                # GPT models use their own API; only start vLLM for local models
                if not eval_server and not eval_model.startswith("gpt"):
                    eval_server = start_vllm_server(eval_model, args.tensor_parallel_size)

                if args.num_workers <= 1:
                    for sample in tqdm(remaining):
                        process_evaluation(sample, eval_file, eval_model)
                else:
                    from functools import partial
                    eval_worker_func = partial(process_evaluation, eval_file=eval_file, model_name=eval_model)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                        futures = [executor.submit(eval_worker_func, sample) for sample in remaining]
                        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            pass

                # Re-check completeness after this pass
                completed_count = len(load_completed_ids(eval_file))
                if completed_count >= total_expected:
                    print(f"All {total_expected} responses evaluated.")
                    break
                else:
                    print(f"Evaluated {completed_count}/{total_expected} responses. Retrying missing ones...")

            print("\nEvaluation completed.")

        finally:
            if eval_server:
                kill_server_process(eval_server)

    # --- Phase 3: Aggregate scores ---
    if args.score:
        print("\nAggregating results...")
        eval_results = load_jsonl(eval_file)
        scores = {}

        # Initialize score accumulators per task
        for task in datasets_to_load:
            if 'mimic' in task:
                scores[task] = {'score_obtained': 0, 'score_total': 0}
            else:
                scores[task] = {'correct': 0, 'incorrect': 0}

        # Accumulate scores from eval results
        for result in eval_results:
            task = "_".join(result.get("id").split('_')[:-1])
            score = result.get("evaluation")['score']
            if 'mimic' in result.get("id"):
                scores[task]['score_obtained'] += float(score)
                scores[task]['score_total'] += 5  # max score per sample
            else:
                if int(score):
                    scores[task]['correct'] += 1
                else:
                    scores[task]['incorrect'] += 1

        # Compute final accuracy per task
        for task in datasets_to_load:
            if 'mimic' in task:
                scores[task] = scores[task]['score_obtained'] / scores[task]['score_total']
            else:
                scores[task] = scores[task]['correct'] / (scores[task]['correct'] + scores[task]['incorrect'])

        # Build results table
        score_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_score.txt")
        data = {'Task': list(scores.keys()), 'Accuracy': list(scores.values())}
        df = pd.DataFrame(data)
        df['Accuracy (%)'] = df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")

        if len(scores) > 1:
            avg_score = sum(scores.values()) / len(scores)
            summary_row = pd.DataFrame({'Task': ['Overall'],
                                    'Accuracy': [avg_score*100],
                                    'Accuracy (%)': [f"{avg_score*100:.2f}%"]})
            df = pd.concat([df, summary_row], ignore_index=True)

        print("\nEvaluation Results:")
        print(df[['Task', 'Accuracy', 'Accuracy (%)']])

        with open(score_file, 'a') as f:
            f.write(f"Model: {model_name}\n\n")
            f.write(df.to_string())

        print(f"\nResults saved to \033[1m{score_file}\033[0m\n")

if __name__ == "__main__":
    main()
