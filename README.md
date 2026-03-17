# LiveMathematicianBench
A live benchmark for evaluating LLM's capability as a mathematician

## Installation

```bash
# Create virtual environment with uv
uv venv .venv --python 3.11

# Install dependencies
uv pip install -r requirements.txt --python .venv/bin/python
```

## Evaluation

```bash
source .venv/bin/activate

python eval/eval.py \
    --model gpt-5.2 \
    --endpoint https://your-endpoint.cognitiveservices.azure.com/ \
    --api-key <your-api-key> \
    --month 202511 \
    --reasoning-effort medium \
    --max-tokens 16384 \
    --concurrency 4
```

### Options

| Flag | Description | Default |
|---|---|---|
| `--model` | Model deployment name (e.g. `gpt-5.2`) | required |
| `--endpoint` | Azure OpenAI endpoint URL | required |
| `--api-key` | Azure OpenAI API key | required |
| `--api-version` | API version | `2024-12-01-preview` |
| `--month` | Month(s) to evaluate (e.g. `202511 202512`) | required |
| `--reasoning-effort` | `low`, `medium`, or `high` | none |
| `--max-tokens` | Max completion tokens | `16384` |
| `--concurrency` | Parallel requests | `4` |
| `--seed` | Random seed for choice shuffling | `42` |
| `--resume` | Resume from previous run, skip answered questions | off |

Results are saved to `results/<month>/accuracy_test_<model>_<month>_<effort>.json`.
