#!/usr/bin/env python3
"""
Evaluate a pychomsky-backed model on the LiveMathematicianBench hard set.

Usage:
    python3 eval/eval_pychomsky.py \
        --model gcp-chat-completions-anthropic-claude-opus-4.6-sandbox \
        --endpoint https://chomskygw6cont.pp.vip.ebay.com/api/v1/genai \
        --month 202511 \
        --reasoning-effort max \
        --max-tokens 128000 \
        --concurrency 4

Results are saved to results/<month>/accuracy_test_<model>_<month>_<reasoning_effort>.json
"""

import argparse
import json
import random
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

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

_CHAT_DEPS = None


def load_chat_dependencies() -> dict:
    global _CHAT_DEPS
    if _CHAT_DEPS is None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from pychomsky.chchat import GCPVertexAnthropicChatWrapper
        except ImportError as exc:
            raise RuntimeError(
                "Missing required dependencies for eval_pychomsky.py. "
                "Install `pychomsky`, `langchain`, and `langchain-core` in the active environment."
            ) from exc

        _CHAT_DEPS = {
            "HumanMessage": HumanMessage,
            "SystemMessage": SystemMessage,
            "ChatWrapper": GCPVertexAnthropicChatWrapper,
        }
    return _CHAT_DEPS


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
    return "\n\n".join(f"({choice['label']}) {choice['text']}" for choice in choices)


def build_prompt(item: dict, choices: list[dict]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=item["mcq"]["question"],
        choices_text=format_choices(choices),
    )


def extract_answer(response_text: str) -> str | None:
    """Extract a single letter answer (A-E) from the model response."""
    text = response_text.strip()
    boxed = re.findall(r"\\boxed\{([A-E])\}", text)
    if boxed:
        return boxed[-1]
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    matches = re.findall(r"\b([A-E])\b", text)
    if matches:
        return matches[-1]
    return None


def stringify_content(content, include_thinking: bool) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            if not isinstance(part, dict):
                pieces.append(str(part))
                continue

            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if text:
                    pieces.append(text)
            elif include_thinking and part_type == "thinking":
                thinking = part.get("thinking")
                if thinking:
                    pieces.append(thinking)
            else:
                fallback = part.get("text")
                if fallback:
                    pieces.append(fallback)
        return "".join(pieces) if pieces else None
    return str(content)


def extract_raw_response(response) -> str | None:
    content = getattr(response, "content", None)
    raw_response = stringify_content(content, include_thinking=False)
    if raw_response:
        return raw_response
    return stringify_content(content, include_thinking=True)


def extract_usage(response) -> tuple[int | None, int | None, int | None, int | None]:
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    reasoning_tokens = None

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = response_metadata.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            reasoning_tokens = usage.get("reasoning_tokens")

            details = usage.get("completion_tokens_details")
            if reasoning_tokens is None and isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        if prompt_tokens is None:
            prompt_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens")
        if completion_tokens is None:
            completion_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens")
        if total_tokens is None:
            total_tokens = usage_metadata.get("total_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = usage_metadata.get("reasoning_tokens")

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens, reasoning_tokens


def map_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort == "xhigh":
        return "max"
    return reasoning_effort


def create_chat_llm(
    model: str,
    chgw_endpoint: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
    thinking_type: str,
):
    deps = load_chat_dependencies()
    model_kwargs = {"thinking": {"type": thinking_type}}
    api_effort = map_reasoning_effort(reasoning_effort)
    if api_effort:
        model_kwargs["output_config"] = {"effort": api_effort}

    return deps["ChatWrapper"](
        chgw_endpoint=chgw_endpoint,
        model_name=model,
        max_output_tokens=max_tokens,
        temperature=temperature,
        model_kwargs=model_kwargs,
    )


def evaluate_single(
    model: str,
    chgw_endpoint: str,
    item: dict,
    seed: int,
    max_tokens: int,
    reasoning_effort: str | None,
    temperature: float,
    thinking_type: str,
    n_samples: int = 1,
) -> dict:
    """Evaluate a single question with n_samples generations. Returns a result dict."""
    deps = load_chat_dependencies()
    choices, correct_label = build_choices(item, seed)
    user_prompt = build_prompt(item, choices)
    messages = [
        deps["SystemMessage"](content=SYSTEM_PROMPT),
        deps["HumanMessage"](content=user_prompt),
    ]

    chat_llm = create_chat_llm(
        model=model,
        chgw_endpoint=chgw_endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        thinking_type=thinking_type,
    )

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
            response = chat_llm.invoke(messages)
            raw_response = extract_raw_response(response)
            model_answer = extract_answer(raw_response or "")
            if model_answer is None:
                full_content = stringify_content(getattr(response, "content", None), include_thinking=True)
                if full_content and full_content != raw_response:
                    model_answer = extract_answer(full_content)

            prompt_tokens, completion_tokens, total_tokens, reasoning_tokens = extract_usage(response)
        except Exception as exc:
            error = str(exc)

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
        description="Evaluate a pychomsky-backed model on LiveMathematicianBench hard set"
    )
    parser.add_argument("--model", required=True, help="Model name for the Chomsky gateway")
    parser.add_argument(
        "--endpoint",
        "--chgw-endpoint",
        dest="chgw_endpoint",
        required=True,
        help="Chomsky gateway endpoint URL",
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
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level (xhigh is mapped to max for the pychomsky API)",
    )
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--thinking-type", default="adaptive", help="Thinking mode to send in model_kwargs")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for choice shuffling")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generations to sample per question (for avg@n, pass@n)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from previous run, skip already answered questions")
    args = parser.parse_args()

    try:
        load_chat_dependencies()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    for month in args.month:
        print(f"\n{'=' * 60}")
        print(f"Evaluating month: {month}")
        print(f"{'=' * 60}")

        data = load_hard_set(month)
        print(f"Loaded {len(data)} questions")

        effort_tag = args.reasoning_effort or "default"
        safe_model = re.sub(r"[^\w\-.]", "_", args.model)
        month_dir = RESULTS_DIR / month
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / f"accuracy_test_{safe_model}_{month}_{effort_tag}.json"

        prev_results = {}
        if args.resume and out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            for result in prev_data.get("detailed_results", []):
                if result.get("model_answer") is not None and result.get("error") is None:
                    prev_results[result["id"]] = result
            print(f"Resuming: {len(prev_results)} already answered, {len(data) - len(prev_results)} remaining")

        results = list(prev_results.values())
        correct = sum(1 for result in results if result["is_correct"])
        total = len(data)
        done = len(results)
        pending = [(i, item) for i, item in enumerate(data) if item["id"] not in prev_results]

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            for i, item in pending:
                per_item_seed = args.seed + i
                future = executor.submit(
                    evaluate_single,
                    args.model,
                    args.chgw_endpoint,
                    item,
                    per_item_seed,
                    args.max_tokens,
                    args.reasoning_effort,
                    args.temperature,
                    args.thinking_type,
                    args.n,
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
                rt = result.get("reasoning_tokens")
                rt_str = f", reasoning_tokens={rt}" if rt is not None else ""
                n_correct = sum(1 for sample in (result.get("samples") or [result]) if sample["is_correct"])
                n_str = f", pass={n_correct}/{args.n}" if args.n > 1 else ""
                print(
                    f"  [{done:3d}/{total}] {status} {item_id}  "
                    f"(answer={result['model_answer']}, correct={result['correct_answer']}, "
                    f"{result['elapsed_seconds']:.1f}s{rt_str}{n_str})"
                )

        id_order = {item["id"]: i for i, item in enumerate(data)}
        results.sort(key=lambda result: id_order.get(result["id"], 0))

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
