#!/usr/bin/env python3
"""
Evaluate an Azure OpenAI model on the LiveMathematicianBench hard set.

Usage:
    python eval/eval.py \
        --model gpt-5.2 \
        --endpoint https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/ \
        --api-key <key> \
        --month 202511 \
        --reasoning-effort medium \
        --max-tokens 16384 \
        --concurrency 4

Results are saved to results/<model>_<month>_<reasoning_effort>.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import AzureOpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

SYSTEM_PROMPT = """\
You are an expert mathematician. You will be given a multiple-choice question. \
Read the question carefully and select the single best answer choice. \
Please reason step by step, and put the letter (A, B, C, D, or E) of your chosen answer in \\boxed{{}}, like this: \\boxed{{A}}. \
Do NOT include any other text in \\boxed{{}}."""

USER_PROMPT_TEMPLATE = """\
## Question

{question}

## Answer Choices

{choices_text}"""


def load_hard_set(month: str) -> list:
    path = DATA_DIR / month / "hard" / f"qaEval_{month}_ge5_hard.json"
    if not path.exists():
        print(f"Error: Hard set not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_choices(item: dict, seed: int) -> tuple[list[dict], str]:
    """Shuffle choices and return (shuffled_choices, correct_label)."""
    correct = item["mcq"]["correct_choice"]
    distractors = item["mcq"]["choices"]
    all_choices = [correct] + distractors

    rng = random.Random(seed)
    rng.shuffle(all_choices)

    labels = ["A", "B", "C", "D", "E"]
    correct_label = None
    labeled = []
    for i, choice in enumerate(all_choices):
        label = labels[i]
        labeled.append({"label": label, "text": choice["text"]})
        if choice["label"] == correct["label"] and choice["text"] == correct["text"]:
            correct_label = label

    return labeled, correct_label


def format_choices(choices: list[dict]) -> str:
    lines = []
    for c in choices:
        lines.append(f"({c['label']}) {c['text']}")
    return "\n\n".join(lines)


def build_prompt(item: dict, choices: list[dict]) -> str:
    question = item["mcq"]["question"]
    choices_text = format_choices(choices)

    return USER_PROMPT_TEMPLATE.format(
        question=question,
        choices_text=choices_text,
    )


def extract_answer(response_text: str) -> str | None:
    """Extract a single letter answer (A-E) from the model response."""
    text = response_text.strip()
    # Look for \boxed{X} first (matches the system prompt format)
    boxed = re.findall(r'\\boxed\{([A-E])\}', text)
    if boxed:
        return boxed[-1]
    # Direct single letter
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    # Fallback: last standalone letter A-E in the response
    m = re.findall(r'\b([A-E])\b', text)
    if m:
        return m[-1]
    return None


def evaluate_single(
    client: AzureOpenAI,
    model: str,
    item: dict,
    seed: int,
    max_tokens: int,
    reasoning_effort: str | None,
    n_samples: int = 1,
    request_timeout: int | None = None,
    use_responses_api: bool = False,
) -> dict:
    """Evaluate a single question with n_samples generations. Returns a result dict."""
    choices, correct_label = build_choices(item, seed)
    user_prompt = build_prompt(item, choices)

    samples = []
    for sample_idx in range(n_samples):
        start = time.time()
        error = None
        model_answer = None
        raw_response = None
        reasoning_tokens = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        try:
            if use_responses_api:
                kwargs = {
                    "model": model,
                    "instructions": SYSTEM_PROMPT,
                    "input": user_prompt,
                    "max_output_tokens": max_tokens,
                }
                if reasoning_effort:
                    kwargs["reasoning"] = {"effort": reasoning_effort}
                if request_timeout:
                    kwargs["timeout"] = request_timeout

                response = client.responses.create(**kwargs)
                raw_response = response.output_text or ""
                model_answer = extract_answer(raw_response)
                if response.usage:
                    prompt_tokens = response.usage.input_tokens
                    completion_tokens = response.usage.output_tokens
                    total_tokens = response.usage.total_tokens
                    api_reasoning = None
                    if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details:
                        api_reasoning = getattr(response.usage.output_tokens_details, 'reasoning_tokens', None)
                    reasoning_tokens = api_reasoning
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": max_tokens,
                }
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                if request_timeout:
                    kwargs["timeout"] = request_timeout

                response = client.chat.completions.create(**kwargs)
                raw_response = response.choices[0].message.content or ""
                model_answer = extract_answer(raw_response)
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    if response.usage.completion_tokens_details:
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        except Exception as e:
            error = str(e)

        elapsed = time.time() - start

        samples.append({
            "sample_idx": sample_idx,
            "model_answer": model_answer,
            "raw_response": raw_response,
            "is_correct": model_answer == correct_label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "elapsed_seconds": elapsed,
            "error": error,
        })

    # For backward compatibility, top-level fields use the first sample
    first = samples[0]
    return {
        "id": item["id"],
        "theorem_type": "all",
        "score": item.get("mcq", {}).get("meta", {}).get("score", None),
        "correct_answer": correct_label,
        "model_answer": first["model_answer"],
        "raw_response": first["raw_response"],
        "is_correct": first["is_correct"],
        "reasoning_effort": reasoning_effort,
        "prompt_tokens": first["prompt_tokens"],
        "completion_tokens": first["completion_tokens"],
        "total_tokens": first["total_tokens"],
        "reasoning_tokens": first["reasoning_tokens"],
        "elapsed_seconds": first["elapsed_seconds"],
        "error": first["error"],
        "n_samples": n_samples,
        "samples": samples if n_samples > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an Azure OpenAI model on LiveMathematicianBench hard set"
    )
    parser.add_argument("--model", required=True, help="Model deployment name (e.g. gpt-5.2)")
    parser.add_argument("--endpoint", required=True, help="Azure OpenAI endpoint URL")
    parser.add_argument("--api-key", required=True, help="Azure OpenAI API key")
    parser.add_argument(
        "--api-version", default="2024-12-01-preview", help="API version (default: 2024-12-01-preview)"
    )
    parser.add_argument(
        "--month",
        required=True,
        nargs="+",
        help="Month(s) to evaluate, e.g. 202511 202512",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort level (optional)",
    )
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max completion tokens")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for choice shuffling")
    parser.add_argument("--n", type=int, default=1, help="Number of generations to sample per question (for avg@n, pass@n)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run, skip already answered questions")
    parser.add_argument("--timeout", type=int, default=7200, help="HTTP client timeout in seconds (default: 7200)")
    parser.add_argument("--request-timeout", type=int, default=3600, help="Per-request timeout in seconds for each sample (default: 3600)")
    parser.add_argument("--use-responses-api", action="store_true", help="Use OpenAI Responses API (client.responses.create) instead of Chat Completions. Auto-enabled for gpt-5.4.")
    args = parser.parse_args()

    # Auto-enable responses API for gpt-5.4
    if "gpt-5.4" in args.model:
        args.use_responses_api = True

    client = AzureOpenAI(
        azure_endpoint=args.endpoint,
        api_key=args.api_key,
        api_version=args.api_version,
        timeout=args.timeout,
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    for month in args.month:
        print(f"\n{'='*60}")
        print(f"Evaluating month: {month}")
        print(f"{'='*60}")

        data = load_hard_set(month)
        print(f"Loaded {len(data)} questions")

        # Resume: load previous results and skip completed questions
        effort_tag = args.reasoning_effort or "default"
        safe_model = re.sub(r'[^\w\-.]', '_', args.model)
        month_dir = RESULTS_DIR / month
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / f"accuracy_test_{safe_model}_{month}_{effort_tag}.json"

        prev_results = {}
        if args.resume and out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            for r in prev_data.get("detailed_results", []):
                if r.get("model_answer") is not None and r.get("error") is None:
                    prev_results[r["id"]] = r
            print(f"Resuming: {len(prev_results)} already answered, {len(data) - len(prev_results)} remaining")

        results = list(prev_results.values())
        correct = sum(1 for r in results if r["is_correct"])
        total = len(data)
        done = len(results)

        # Filter to only unanswered items
        pending = [(i, item) for i, item in enumerate(data) if item["id"] not in prev_results]

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            for i, item in pending:
                per_item_seed = args.seed + i
                future = executor.submit(
                    evaluate_single,
                    client,
                    args.model,
                    item,
                    per_item_seed,
                    args.max_tokens,
                    args.reasoning_effort,
                    args.n,
                    args.request_timeout,
                    args.use_responses_api,
                )
                futures[future] = item["id"]

            for future in as_completed(futures):
                item_id = futures[future]
                result = future.result()
                results.append(result)
                if result["is_correct"]:
                    correct += 1
                done += 1
                status = "✓" if result["is_correct"] else "✗"
                rt = result.get("reasoning_tokens", None)
                rt_str = f", reasoning_tokens={rt}" if rt is not None else ""
                n_correct = sum(1 for s in (result.get("samples") or [result]) if s["is_correct"])
                n_str = f", pass={n_correct}/{args.n}" if args.n > 1 else ""
                print(
                    f"  [{done:3d}/{total}] {status} {item_id}  "
                    f"(answer={result['model_answer']}, correct={result['correct_answer']}, "
                    f"{result['elapsed_seconds']:.1f}s{rt_str}{n_str})"
                )

        # Sort results by original order (by matching ids)
        id_order = {item["id"]: i for i, item in enumerate(data)}
        results.sort(key=lambda r: id_order.get(r["id"], 0))

        accuracy = correct / total if total > 0 else 0.0

        output = {
            "test_info": {
                "date": datetime.now().strftime("%m%d_%H%M%S"),
                "model": args.model,
                "month": month,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "n_samples": args.n,
                "total_candidates": total,
                "total_tested": total,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": {
                "all": {
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                }
            },
            "overall": {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            },
            "detailed_results": results,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults: {correct}/{total} = {accuracy:.2%}")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
