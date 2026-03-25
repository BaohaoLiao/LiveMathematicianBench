#!/usr/bin/env python3
"""
Build script for LiveMathematicianBench website.
Reads all JSON data files and generates a self-contained docs/index.html.
"""
import json
import os
import glob
import html
import re
import random
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("docs")
MONTHS = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
CATEGORIES = [
    "Asymptotic or Limit",
    "Biconditional or Equivalence",
    "Classification or Bijection",
    "Existence",
    "Existential–Universal",
    "Implication",
    "Inequality or Bound",
    "Uniqueness",
    "Universal",
    "Other",
]
MONTH_LABELS = {
    "202511": "Nov 2025",
    "202512": "Dec 2025",
    "202601": "Jan 2026",
    "202602": "Feb 2026",
    "202603": "Mar 2026",
    "202604": "Apr 2026",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hard_data(month):
    """Load hard QA data for a given month."""
    fpath = DATA_DIR / month / "hard" / f"qaEval_{month}_ge5_hard.json"
    if fpath.exists():
        return load_json(fpath)
    return []


def collect_stats():
    """Collect per-month, per-category question counts from hard data."""
    stats = {}
    for month in MONTHS:
        stats[month] = {}
        for cat in CATEGORIES:
            stats[month][cat] = 0
        data = load_hard_data(month)
        for item in data:
            for t in item.get("theorem_type", []):
                if t in CATEGORIES:
                    stats[month][t] += 1
        # Also store total hard questions (an item may span multiple categories)
        stats[month]["_total"] = len(data)
    return stats


def collect_examples():
    """Collect example questions from hard data for each month."""
    examples = []
    seen_ids = set()
    for month in MONTHS:
        data = load_hard_data(month)
        for item in data:
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            types = item.get("theorem_type", [])
            cat = next((t for t in types if t in CATEGORIES), types[0] if types else "Other")
            correct_choice = item.get("mcq", {}).get("correct_choice", {})
            wrong_choices = item.get("mcq", {}).get("choices", [])
            all_choices = [{"text": correct_choice.get("text", ""), "correct": True}] + \
                          [{"text": ch.get("text", ""), "correct": False} for ch in wrong_choices]
            rng = random.Random(42 + hash(item["id"]))
            rng.shuffle(all_choices)
            labels = "ABCDE"
            labeled_choices = [{"label": labels[i], "text": ch["text"], "correct": ch["correct"]} for i, ch in enumerate(all_choices)]
            examples.append({
                "id": item["id"],
                "month": month,
                "category": cat,
                "paper_link": item.get("paper_link", ""),
                "theorem": re.sub(r'\\\\?label\{[^}]*\}\s*', '', item.get("expanded_theorem", item.get("theorem", {}).get("content", ""))).strip(),
                "sketch": item.get("expanded_sketch", ""),
                "question": item.get("mcq", {}).get("question", ""),
                "choices": labeled_choices,
            })
    return examples


def _build_category_map():
    """Build a map from question id to its theorem categories across all months."""
    cat_map = {}
    for month in MONTHS:
        for item in load_hard_data(month):
            cat_map[item["id"]] = [t for t in item.get("theorem_type", []) if t in CATEGORIES]
    return cat_map


def collect_accuracy():
    """Collect model accuracy results from hard accuracy_test files in data/ and results/."""
    cat_map = _build_category_map()
    results = []
    seen = set()
    search_patterns = [
        str(DATA_DIR / "*/hard/accuracy_test_*.json"),
        str(RESULTS_DIR / "*/accuracy_test_*.json"),
    ]
    for pattern in search_patterns:
        for fpath in sorted(glob.glob(pattern)):
            # Skip .progress files (incomplete runs)
            if ".progress." in fpath:
                continue
            # Skip sketch result files (those are for the sketch comparison section)
            if "_sketch" in os.path.basename(fpath):
                continue
            data = load_json(fpath)
            ti = data.get("test_info", {})
            # Deduplicate by (model, month, reasoning_effort)
            key = (ti.get("model", ""), ti.get("month", ""), ti.get("reasoning_effort", ""))
            if key in seen:
                continue
            seen.add(key)

            # Compute per-category accuracy from detailed_results
            cat_correct = {c: 0 for c in CATEGORIES}
            cat_total = {c: 0 for c in CATEGORIES}
            for r in data.get("detailed_results", []):
                cats = cat_map.get(r.get("id", ""), [])
                for c in cats:
                    cat_total[c] += 1
                    if r.get("is_correct"):
                        cat_correct[c] += 1
            category_accuracy = {}
            for c in CATEGORIES:
                if cat_total[c] > 0:
                    category_accuracy[c] = {
                        "correct": cat_correct[c],
                        "total": cat_total[c],
                        "accuracy": cat_correct[c] / cat_total[c],
                    }

            # Compute average output tokens (total - prompt) and elapsed time from detailed results (only for samples with a valid answer)
            answered = [r for r in data.get("detailed_results", []) if r.get("model_answer") is not None]
            comp_tokens = [r.get("total_tokens", 0) - r.get("prompt_tokens", 0) for r in answered if r.get("total_tokens") is not None and r.get("prompt_tokens") is not None]
            avg_completion_tokens = sum(comp_tokens) / len(comp_tokens) if comp_tokens else None
            elapsed = [r.get("elapsed_seconds", 0) for r in answered if r.get("elapsed_seconds") is not None]
            avg_elapsed_seconds = sum(elapsed) / len(elapsed) if elapsed else None

            model = ti.get("model", "")
            reasoning_effort = ti.get("reasoning_effort", "")
            # Models with built-in reasoning that don't use the reasoning_effort API param
            if not reasoning_effort and any(k in model for k in ["Qwen3.5-397B-A17B", "Kimi-K2.5", "MiniMax-M2.5"]):
                reasoning_effort = "enabled"

            results.append({
                "file": os.path.basename(fpath),
                "month": ti.get("month", ""),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "overall": data.get("overall", {}),
                "summary": data.get("summary", {}),
                "category_accuracy": category_accuracy,
                "avg_completion_tokens": avg_completion_tokens,
                "avg_elapsed_seconds": avg_elapsed_seconds,
            })
    return results


def collect_sketch_accuracy():
    """Collect sketch result files (ending with _sketch.json) for comparison."""
    results = []
    seen = set()
    search_patterns = [
        str(DATA_DIR / "*/hard/accuracy_test_*_sketch.json"),
        str(RESULTS_DIR / "*/accuracy_test_*_sketch.json"),
    ]
    for pattern in search_patterns:
        for fpath in sorted(glob.glob(pattern)):
            if ".progress." in fpath:
                continue
            data = load_json(fpath)
            ti = data.get("test_info", {})
            key = (ti.get("model", ""), ti.get("month", ""), ti.get("reasoning_effort", ""))
            if key in seen:
                continue
            seen.add(key)

            model = ti.get("model", "")
            reasoning_effort = ti.get("reasoning_effort", "")
            if not reasoning_effort and any(k in model for k in ["Qwen3.5-397B-A17B", "Kimi-K2.5", "MiniMax-M2.5"]):
                reasoning_effort = "enabled"

            results.append({
                "file": os.path.basename(fpath),
                "month": ti.get("month", ""),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "overall": data.get("overall", {}),
            })
    return results


def build():
    stats = collect_stats()
    examples = collect_examples()
    accuracy = collect_accuracy()
    sketch_accuracy = collect_sketch_accuracy()

    # JSON-encode data for embedding
    stats_json = json.dumps(stats)
    examples_json = json.dumps(examples)
    accuracy_json = json.dumps(accuracy)
    sketch_accuracy_json = json.dumps(sketch_accuracy)
    months_json = json.dumps(MONTHS)
    categories_json = json.dumps(CATEGORIES)
    month_labels_json = json.dumps(MONTH_LABELS)

    OUTPUT_DIR.mkdir(exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LiveMathematicianBench</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --primary: #1a56db;
    --primary-light: #e1effe;
    --accent: #7c3aed;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
    --table-header-bg: #f1f5f9;
    --content-bg: #f8fafc;
    --hover-bg: #f8fafc;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}

  /* Hero */
  .hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    color: white;
    padding: 80px 0 60px;
    text-align: center;
  }}
  .hero h1 {{ font-size: 2.8rem; font-weight: 800; margin-bottom: 12px; letter-spacing: -0.02em; }}
  .hero .subtitle {{ font-size: 1.25rem; opacity: 0.9; max-width: 700px; margin: 0 auto 24px; }}
  .hero .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.9rem;
    backdrop-filter: blur(4px);
  }}
  .hero-links {{ margin-top: 20px; }}
  .hero-links a {{
    color: white;
    text-decoration: none;
    margin: 0 12px;
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.2s;
  }}
  .hero-links a.btn-primary {{ background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); }}
  .hero-links a.btn-primary:hover {{ background: rgba(255,255,255,0.3); }}

  /* Navigation tabs */
  .nav-tabs {{
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .nav-tabs .container {{ display: flex; gap: 0; overflow-x: auto; }}
  .nav-tab {{
    padding: 14px 20px;
    cursor: pointer;
    font-weight: 500;
    color: var(--text-muted);
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: all 0.2s;
    user-select: none;
  }}
  .nav-tab:hover {{ color: var(--primary); }}
  .nav-tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}

  /* Sections */
  .section {{ padding: 48px 0; display: none; }}
  .section.active {{ display: block; }}
  .section-title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }}
  .section-desc {{ color: var(--text-muted); margin-bottom: 32px; font-size: 1.05rem; }}

  /* Cards */
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .card h3 {{ font-size: 1.1rem; margin-bottom: 12px; }}

  /* Stat boxes */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .stat-box {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .stat-box .stat-value {{
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1.2;
  }}
  .stat-box .stat-label {{
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
  }}

  /* Tables */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: var(--table-header-bg);
    font-weight: 700;
    color: var(--text);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  #stats-table {{ font-size: 1rem; }}
  #stats-table th {{ text-align: center; background: #fff; }}
  #stats-table td {{ text-align: center; }}
  #stats-table th:first-child,
  #stats-table td:first-child {{ text-align: left; }}
  tr:hover td {{ background: var(--hover-bg); }}
  #lb-overall-table {{ font-size: 0.85rem; }}
  #lb-overall-table th {{ text-align: center; background: #fff; }}
  #lb-overall-table td {{ text-align: center; }}
  #lb-overall-table th:nth-child(3),
  #lb-overall-table td:nth-child(3) {{ text-align: left; }}
  #lb-monthly-table {{ font-size: 0.85rem; }}
  #lb-monthly-table th {{ text-align: center; background: #fff; }}
  #lb-monthly-table td {{ text-align: center; }}
  #lb-monthly-table th:nth-child(3),
  #lb-monthly-table td:nth-child(3) {{ text-align: left; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  /* Charts */
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 32px;
  }}
  @media (max-width: 768px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
    .hero h1 {{ font-size: 2rem; }}
  }}
  .chart-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 12px; color: var(--text); font-weight: 700; }}
  .chart-container {{ position: relative; height: 300px; }}

  /* Examples */
  .example-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
  }}
  .example-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
  .example-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .example-header .tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .tag-cat {{ background: var(--primary-light); color: var(--primary); }}
  .tag-month {{ background: #fef3c7; color: #92400e; }}
  .example-header a {{
    color: var(--primary);
    font-size: 0.85rem;
    text-decoration: none;
    margin-left: auto;
  }}
  .example-header a:hover {{ text-decoration: underline; }}
  .example-section {{ margin-bottom: 14px; }}
  .example-section .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .example-section .content {{
    background: var(--content-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.95rem;
    overflow-x: auto;
  }}
  .choices-list {{ list-style: none; padding: 0; }}
  .choices-list li {{
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 8px;
    background: var(--content-bg);
    font-size: 0.9rem;
    border: 1px solid var(--border);
  }}
  .choices-list li.correct {{
    background: #ecfdf5;
    border-color: #a7f3d0;
  }}
  .choices-list li .choice-label {{
    font-weight: 700;
    margin-right: 8px;
    color: var(--text-muted);
  }}
  .choices-list li.correct .choice-label {{ color: var(--success); }}

  /* Filters */
  .filter-bar {{
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-bar select, .filter-bar input {{
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.9rem;
    background: var(--card-bg);
    color: var(--text);
  }}

  /* Leaderboard */
  .leaderboard-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 24px;
  }}
  .leaderboard-card h3 {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 1rem;
    color: var(--text);
    font-weight: 700;
    background: var(--table-header-bg);
  }}
  .accuracy-bar-wrap {{
    display: inline-block;
    width: 80px;
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    vertical-align: middle;
    margin-left: 8px;
    position: relative;
    overflow: hidden;
  }}
  .accuracy-bar {{
    height: 100%;
    border-radius: 4px;
    background: var(--primary);
    min-width: 2px;
  }}
  .accuracy-value {{
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .pct-high {{ color: var(--success); }}
  .pct-mid {{ color: var(--warning); }}
  .pct-low {{ color: var(--danger); }}

  /* Pagination for examples */
  .pagination {{
    display: flex;
    gap: 8px;
    justify-content: center;
    margin: 24px 0;
  }}
  .pagination button {{
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
  }}
  .pagination button:hover {{ background: var(--primary-light); }}
  .pagination button.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
  .pagination button:disabled {{ opacity: 0.4; cursor: not-allowed; }}

  /* About section */
  .about-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 768px) {{ .about-grid {{ grid-template-columns: 1fr; }} }}
  .about-card {{ padding: 24px; }}
  .about-card h3 {{ color: var(--primary); margin-bottom: 8px; }}
  .about-card p {{ color: var(--text-muted); font-size: 0.95rem; }}
  .about-card ul {{ color: var(--text-muted); font-size: 0.95rem; margin-left: 20px; margin-top: 8px; }}

  /* Footer */
  footer {{
    text-align: center;
    padding: 32px 0;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}
  footer a {{ color: var(--primary); text-decoration: none; }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <div class="container">
    <div class="badge">&#x1f4d0; Live Benchmark &mdash; Updated Regularly</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:20px;margin:16px 0;">
      <img src="lmb_logo.svg" alt="LiveMathematicianBench" style="height:80px;width:80px;border-radius:50%;object-fit:cover;">
      <h1 style="margin:0;"><span style="color:#60a5fa;">Live</span>MathematicianBench</h1>
    </div>
    <p class="subtitle">
      A live benchmark for evaluating LLMs' capability as mathematicians,
      featuring research-level theorem comprehension from the latest arXiv papers.
    </p>
    <div class="hero-links">
      <a class="btn-primary" href="https://github.com/BaohaoLiao/LiveMathematicianBench" target="_blank" rel="noopener"><svg height="16" width="16" viewBox="0 0 16 16" fill="white" style="vertical-align:-2px;margin-right:6px;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>GitHub</a>
      <a class="btn-primary" href="https://huggingface.co/datasets/hendrydong/livemath-v7-0316" target="_blank" rel="noopener"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" height="16" style="vertical-align:-2px;margin-right:6px;">Dataset</a>
    </div>
    <p style="margin-top:12px;font-size:0.85rem;color:rgba(255,255,255,0.6);">Last updated: {datetime.now().strftime('%B %d, %Y')}</p>
  </div>
</div>

<!-- Nav -->
<div class="nav-tabs">
  <div class="container">
    <div class="nav-tab active" data-tab="leaderboard">Leaderboard</div>
    <div class="nav-tab" data-tab="overview">Overview</div>
    <div class="nav-tab" data-tab="examples">Tasks</div>
    <div class="nav-tab" data-tab="about">About</div>
  </div>
</div>

<!-- Sections -->
<div class="container">

  <!-- LEADERBOARD -->
  <div class="section active" id="sec-leaderboard">
    <h2 class="section-title">Model Leaderboard</h2>
    <div id="leaderboard-overall" class="leaderboard-card">
      <h3 style="background:#fff;">Overall Accuracy
        <span style="margin-left:12px;font-size:0.85rem;font-weight:400;">
          <select id="lb-month-from" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
          <span style="margin:0 4px;">&ndash;</span>
          <select id="lb-month-to" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
        </span>
      </h3>
      <div class="table-wrap"><table id="lb-overall-table"></table></div>
    </div>
    <div id="leaderboard-monthly" class="leaderboard-card">
      <h3 style="background:#fff;">Accuracy by Month
        <span style="margin-left:12px;font-size:0.85rem;font-weight:400;">
          <select id="lb-monthly-from" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
          <span style="margin:0 4px;">&ndash;</span>
          <select id="lb-monthly-to" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
        </span>
      </h3>
      <div class="table-wrap"><table id="lb-monthly-table"></table></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card" style="grid-column: 1 / -1;">
        <h3>Accuracy by Category
          <select id="radar-month-filter" style="margin-left:12px;font-size:0.85rem;font-weight:400;padding:4px 8px;border-radius:6px;border:1px solid var(--border);">
            <option value="">All Months</option>
          </select>
          <span style="font-weight:400;font-size:0.8rem;color:var(--text-muted);margin-left:8px;">Click legend to hide/show a model</span>
        </h3>
        <div style="text-align:right;margin-bottom:4px;"><button id="radar-toggle-all" style="font-size:0.78rem;font-weight:500;padding:3px 10px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--text);cursor:pointer;">Hide All</button></div>
        <div class="chart-container" style="height:480px;"><canvas id="chart-model-cat"></canvas></div>
      </div>
    </div>
    <div id="leaderboard-sketch" class="leaderboard-card">
      <h3 style="background:#fff;">Sketch Comparison
        <span style="margin-left:12px;font-size:0.85rem;font-weight:400;">
          <select id="sketch-month-from" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
          <span style="margin:0 4px;">&ndash;</span>
          <select id="sketch-month-to" style="font-size:0.85rem;padding:4px 8px;border-radius:6px;border:1px solid var(--border);"></select>
        </span>
        <span style="font-weight:400;font-size:0.8rem;color:var(--text-muted);margin-left:8px;">Performance with vs. without proof sketch</span>
      </h3>
      <div class="chart-container" style="height:400px;padding:16px;"><canvas id="chart-sketch"></canvas></div>
      <div id="sketch-empty" style="display:none;text-align:center;padding:32px;color:var(--text-muted);">No sketch comparison data available for the selected months.</div>
    </div>
  </div>

  <!-- OVERVIEW -->
  <div class="section" id="sec-overview">
    <h2 class="section-title">Benchmark Overview</h2>
    <p class="section-desc">Monthly updated dataset of research-level mathematics MCQs derived from recent arXiv publications.</p>
    <div class="stat-grid" id="overview-stats"></div>
    <div class="chart-grid">
      <div class="chart-card">
        <h3>Questions per Month</h3>
        <div class="chart-container"><canvas id="chart-monthly"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Category Distribution (All Months)</h3>
        <div class="chart-container"><canvas id="chart-categories"></canvas></div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card" style="grid-column: 1 / -1;">
        <h3>Category Breakdown by Month <span style="font-weight:400;font-size:0.8rem;color:var(--text-muted);margin-left:8px;">Note: One question might have different categories at the same time.</span></h3>
        <div class="chart-container" style="height:360px;"><canvas id="chart-cat-monthly"></canvas></div>
      </div>
    </div>
    <div class="leaderboard-card">
      <h3 style="background:#fff;">Detailed Statistics</h3>
      <div class="table-wrap">
      <table id="stats-table">
        <thead><tr><th>Category</th></tr></thead>
        <tbody></tbody>
      </table>
      </div>
    </div>
    <p style="color:var(--text-muted);font-size:0.85rem;margin-top:8px;font-style:italic;">*: One question might have different categories at the same time.</p>
  </div>

  <!-- EXAMPLES -->
  <div class="section" id="sec-examples">
    <h2 class="section-title">Tasks</h2>
    <p class="section-desc">Browse MCQs from the benchmark. Each question is derived from a real arXiv paper theorem.</p>
    <div class="filter-bar">
      <select id="filter-month"><option value="">All Months</option></select>
      <select id="filter-cat"><option value="">All Categories</option></select>
    </div>
    <div id="examples-container"></div>
    <div class="pagination" id="examples-pagination"></div>
  </div>

  <!-- ABOUT -->
  <div class="section" id="sec-about">
    <h2 class="section-title">About the Benchmark</h2>
    <p class="section-desc">Understanding the design and methodology behind LiveMathematicianBench.</p>
    <div class="about-grid">
      <div class="card about-card">
        <h3>What is it?</h3>
        <p>LiveMathematicianBench is a <strong>live, continuously updated</strong> benchmark
           that evaluates LLMs on their ability to understand and reason about
           cutting-edge mathematical theorems from newly published arXiv preprints.</p>
      </div>
      <div class="card about-card">
        <h3>Why "Live"?</h3>
        <p>New papers appear on arXiv every month. We extract theorems from these papers
           and generate multiple-choice questions that test deep mathematical understanding,
           ensuring that models cannot rely on memorized training data.</p>
      </div>
      <div class="card about-card">
        <h3>Question Format</h3>
        <p>Questions and choices are constructed from theorem statements and proof sketches
           extracted from arXiv papers. Each question has five carefully crafted choices
           (one correct, one weaker-but-true, and three false).
           <strong>Only the question and choices are used as input for the model</strong>&mdash;the
           original theorem and proof sketch are not provided.</p>
      </div>
      <div class="card about-card">
        <h3>Theorem Categories</h3>
        <ul>
          <li>Asymptotic or Limit</li>
          <li>Biconditional or Equivalence</li>
          <li>Classification or Bijection</li>
          <li>Existence</li>
          <li>Existential&ndash;Universal</li>
          <li>Implication</li>
          <li>Inequality or Bound</li>
          <li>Uniqueness</li>
          <li>Universal</li>
          <li>Other</li>
        </ul>
      </div>
    </div>
  </div>

</div>

<footer>
  <div class="container">
    LiveMathematicianBench &copy; 2025&ndash;2026 &middot;
    <a href="https://github.com/BaohaoLiao/LiveMathematicianBench" target="_blank" rel="noopener">GitHub</a> &middot;
    <a href="https://huggingface.co/datasets/hendrydong/livemath-v7-0316" target="_blank" rel="noopener">HuggingFace</a>
  </div>
</footer>

<script>
Chart.defaults.color = '#1e293b';
Chart.defaults.borderColor = 'rgba(0,0,0,0.1)';

// ===== Embedded Data =====
const MONTHS = {months_json};
const CATEGORIES = {categories_json};
const MONTH_LABELS = {month_labels_json};
const STATS = {stats_json};
const EXAMPLES = {examples_json};
const ACCURACY = {accuracy_json};
const SKETCH_ACCURACY = {sketch_accuracy_json};

const CAT_COLORS = [
  '#1e3a8a','#2563eb','#60a5fa','#bfdbfe',
  '#9d174d','#ec4899','#f9a8d4',
  '#92400e','#d97706','#fbbf24'
];
function hexToRgba(hex, alpha) {{ const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16); return `rgba(${{r}},${{g}},${{b}},${{alpha}})`; }}
const CAT_COLORS_LIGHT = CAT_COLORS.map(c => hexToRgba(c, 0.15));

function ml(m) {{ return MONTH_LABELS[m] || m; }}
function pctClass(v) {{ return v >= 0.5 ? 'pct-high' : v > 0.2 ? 'pct-mid' : 'pct-low'; }}
function displayModel(name) {{ return name.replace(/_\\d{{4}}-\\d{{2}}-\\d{{2}}/, '').replace(/^random$/i, 'Random').replace(/^gpt-oss-120b$/i, 'GPT-OSS-120B').replace(/^gpt-/i, 'GPT-').replace(/^grok-4-1-fast-reasoning$/i, 'Grok-4.1 Fast Reasoning').replace(/^gemini-3\\.1-pro-preview$/i, 'Gemini 3.1 Pro Preview').replace(/^claude-sonnet-4\\.6$/i, 'Claude Sonnet 4.6').replace(/^claude-opus-4\\.6$/i, 'Claude Opus 4.6').replace(/^moonshotai\//i, '').replace(/^MiniMaxAI\//i, '').replace(/^Qwen\//i, ''); }}
function providerLogo(model) {{
  const m = model.toLowerCase();
  const providers = [
    {{ match: ['gpt', 'o1-', 'o3', 'o4'], name: 'OpenAI', logo: 'logos/openai.png', size: 30 }},
    {{ match: ['claude'], name: 'Anthropic', logo: 'logos/anthropic.png' }},
    {{ match: ['gemini'], name: 'Google Gemini', logo: 'logos/gemini.png' }},
    {{ match: ['grok'], name: 'Grok', logo: 'logos/grok.png' }},
    {{ match: ['llama'], name: 'Meta', logo: 'logos/meta.png' }},
    {{ match: ['mistral'], name: 'Mistral', logo: 'logos/mistral.png' }},
    {{ match: ['deepseek'], name: 'DeepSeek', logo: 'logos/deepseek.png' }},
    {{ match: ['qwen'], name: 'Qwen', logo: 'logos/qwen.png' }},
    {{ match: ['kimi'], name: 'Kimi', logo: 'logos/kimi.png' }},
    {{ match: ['minimax'], name: 'MiniMax', logo: 'logos/minimax.png', size: 26 }},
  ];
  for (const p of providers) {{
    const sz = p.size || 22;
    if (p.match.some(k => m.includes(k))) return '<img src="' + p.logo + '" alt="' + p.name + '" title="' + p.name + '" style="height:' + sz + 'px;width:' + sz + 'px;vertical-align:middle;">';
  }}
  if (m === 'random') return '<span style="font-size:18px;vertical-align:middle;">🎲</span>';
  return '';
}}

// ===== Tab Navigation =====
document.querySelectorAll('.nav-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('sec-' + tab.dataset.tab).classList.add('active');
  }});
}});

// ===== Overview Stats =====
(function() {{
  const totalAll = MONTHS.reduce((s,m) => s + (STATS[m]['_total']||0), 0);
  const nMonths = MONTHS.length;
  const el = document.getElementById('overview-stats');
  el.innerHTML = `
    <div class="stat-box"><div class="stat-value">${{totalAll}}</div><div class="stat-label">Total Questions</div></div>
    <div class="stat-box"><div class="stat-value">${{nMonths}}</div><div class="stat-label">Monthly Snapshots</div></div>
    <div class="stat-box"><div class="stat-value">${{CATEGORIES.length}}</div><div class="stat-label">Theorem Categories</div></div>
  `;
}})();

// ===== Overview Charts =====
(function() {{
  const monthTotals = MONTHS.map(m => STATS[m]['_total']||0);

  new Chart(document.getElementById('chart-monthly'), {{
    type: 'bar',
    data: {{
      labels: MONTHS.map(ml),
      datasets: [{{ label: 'Questions', data: monthTotals, backgroundColor: hexToRgba('#3b82f6', 0.15), borderColor: '#3b82f6', borderWidth: 3, borderRadius: 6 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#1e293b', font: {{ size: 14 }} }}, grid: {{ display: false }} }},
        y: {{ beginAtZero: true, ticks: {{ precision: 0, color: '#1e293b', font: {{ size: 14 }} }}, grid: {{ display: false }} }}
      }}
    }}
  }});

  const catTotals = CATEGORIES.map(c => MONTHS.reduce((a,m) => a + (STATS[m][c]||0), 0));
  new Chart(document.getElementById('chart-categories'), {{
    type: 'doughnut',
    data: {{
      labels: CATEGORIES,
      datasets: [{{ data: catTotals, backgroundColor: CAT_COLORS_LIGHT, borderColor: CAT_COLORS, borderWidth: 3 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, font: {{ size: 14 }}, color: '#1e293b' }} }} }}
    }}
  }});

  new Chart(document.getElementById('chart-cat-monthly'), {{
    type: 'bar',
    data: {{
      labels: MONTHS.map(ml),
      datasets: CATEGORIES.map((c,i) => ({{
        label: c, data: MONTHS.map(m => STATS[m][c]||0),
        backgroundColor: CAT_COLORS_LIGHT[i], borderColor: CAT_COLORS[i], borderWidth: 3, borderRadius: 3
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, font: {{ size: 14 }}, color: '#1e293b' }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#1e293b', font: {{ size: 14 }} }}, grid: {{ display: false }} }},
        y: {{ stacked: true, beginAtZero: true, ticks: {{ precision: 0, color: '#1e293b', font: {{ size: 14 }} }}, grid: {{ display: false }} }}
      }}
    }}
  }});
}})();

// ===== Statistics Table =====
(function() {{
  const thead = document.querySelector('#stats-table thead tr');
  MONTHS.forEach(m => {{ const th = document.createElement('th'); th.textContent = ml(m); thead.appendChild(th); }});
  const thTotal = document.createElement('th'); thTotal.textContent = 'Total'; thead.appendChild(thTotal);

  const tbody = document.querySelector('#stats-table tbody');
  CATEGORIES.forEach(cat => {{
    const tr = document.createElement('tr');
    const tdCat = document.createElement('td'); tdCat.textContent = cat; tr.appendChild(tdCat);
    let rowTotal = 0;
    MONTHS.forEach(m => {{
      const v = STATS[m][cat]||0; rowTotal += v;
      const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
    }});
    const tdT = document.createElement('td'); tdT.textContent = rowTotal; tdT.style.fontWeight = '700'; tr.appendChild(tdT);
    tbody.appendChild(tr);
  }});

  // Total row
  const trT = document.createElement('tr'); trT.style.fontWeight = '700'; trT.style.borderTop = '2px solid var(--border)';
  const tdL = document.createElement('td'); tdL.textContent = 'Total*'; trT.appendChild(tdL);
  let grandTotal = 0;
  MONTHS.forEach(m => {{
    const v = STATS[m]['_total']||0; grandTotal += v;
    const td = document.createElement('td'); td.textContent = v; trT.appendChild(td);
  }});
  const tdGT = document.createElement('td'); tdGT.textContent = grandTotal; trT.appendChild(tdGT);
  tbody.appendChild(trT);
}})();

// ===== Examples =====
(function() {{
  const PER_PAGE = 5;
  let filteredExamples = [...EXAMPLES];
  let currentPage = 0;

  const monthSel = document.getElementById('filter-month');
  const catSel = document.getElementById('filter-cat');
  MONTHS.forEach(m => {{ const o = document.createElement('option'); o.value = m; o.textContent = ml(m); monthSel.appendChild(o); }});
  CATEGORIES.forEach(c => {{ const o = document.createElement('option'); o.value = c; o.textContent = c; catSel.appendChild(o); }});

  function escapeHtml(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

  function renderExamples() {{
    const container = document.getElementById('examples-container');
    const pagEl = document.getElementById('examples-pagination');
    const start = currentPage * PER_PAGE;
    const page = filteredExamples.slice(start, start + PER_PAGE);
    const totalPages = Math.ceil(filteredExamples.length / PER_PAGE);

    container.innerHTML = page.map((ex, idx) => `
      <div class="example-card" data-ex-idx="${{idx}}">
        <div class="example-header">
          <span class="tag tag-month">${{ml(ex.month)}}</span>
          <span class="tag tag-cat">${{escapeHtml(ex.category)}}</span>
          <span style="color:var(--text-muted);font-size:0.85rem;">${{escapeHtml(ex.id)}}</span>
          ${{ex.paper_link ? `<a href="${{escapeHtml(ex.paper_link)}}" target="_blank" rel="noopener">arXiv &#x2197;</a>` : ''}}
        </div>
        <div class="example-section">
          <div class="label">Theorem</div>
          <div class="content math-render" data-field="theorem"></div>
        </div>
        ${{ex.sketch ? `<div class="example-section">
          <div class="label">Proof Sketch</div>
          <div class="content math-render" data-field="sketch"></div>
        </div>` : ''}}
        <div class="example-section">
          <div class="label">Question</div>
          <div class="content math-render" data-field="question"></div>
        </div>
        <div class="example-section">
          <div class="label">Choices</div>
          <ul class="choices-list">
            ${{ex.choices.map((ch, ci) => `<li class="${{ch.correct ? 'correct' : ''}}"><span class="choice-label">${{escapeHtml(ch.label)}}</span><span class="math-render" data-field="choice-${{ci}}"></span>${{ch.correct ? ' &#x2705;' : ''}}</li>`).join('')}}
          </ul>
        </div>
      </div>
    `).join('');

    // Set math text via textContent to preserve LaTeX characters
    page.forEach((ex, idx) => {{
      const card = container.querySelector(`[data-ex-idx="${{idx}}"]`);
      const setField = (field, text) => {{ const el = card.querySelector(`[data-field="${{field}}"]`); if (el) el.textContent = text; }};
      setField('theorem', ex.theorem);
      if (ex.sketch) setField('sketch', ex.sketch);
      setField('question', ex.question);
      ex.choices.forEach((ch, ci) => setField('choice-' + ci, ch.text));
    }});

    // Pagination
    let pagHtml = `<button ${{currentPage === 0 ? 'disabled' : ''}} onclick="window.__exPage(${{currentPage-1}})">&laquo; Prev</button>`;
    for (let i = 0; i < totalPages; i++) {{
      if (totalPages > 7 && Math.abs(i - currentPage) > 2 && i !== 0 && i !== totalPages-1) {{
        if (i === 1 || i === totalPages - 2) pagHtml += '<button disabled>...</button>';
        continue;
      }}
      pagHtml += `<button class="${{i===currentPage?'active':''}}" onclick="window.__exPage(${{i}})">${{i+1}}</button>`;
    }}
    pagHtml += `<button ${{currentPage >= totalPages-1 ? 'disabled' : ''}} onclick="window.__exPage(${{currentPage+1}})">Next &raquo;</button>`;
    pagEl.innerHTML = pagHtml;

    // Render LaTeX
    container.querySelectorAll('.math-render').forEach(el => {{
      renderMathInElement(el, {{
        delimiters: [
          {{ left: '$$', right: '$$', display: true }},
          {{ left: '\\\\[', right: '\\\\]', display: true }},
          {{ left: '$', right: '$', display: false }},
          {{ left: '\\\\(', right: '\\\\)', display: false }},
        ],
        throwOnError: false,
        trust: true
      }});
    }});
  }}

  window.__exPage = function(p) {{ currentPage = p; renderExamples(); window.scrollTo(0, document.getElementById('sec-examples').offsetTop + 100); }};

  function applyFilters() {{
    const fm = monthSel.value, fc = catSel.value;
    filteredExamples = EXAMPLES.filter(e => (!fm || e.month === fm) && (!fc || e.category === fc));
    currentPage = 0;
    renderExamples();
  }}
  monthSel.addEventListener('change', applyFilters);
  catSel.addEventListener('change', applyFilters);
  renderExamples();
}})();

// ===== Leaderboard =====
(function() {{
  if (ACCURACY.length === 0) {{
    document.getElementById('sec-leaderboard').querySelector('.section-desc').textContent += ' No model evaluations available yet.';
    return;
  }}

  // Overall table with month filter
  const tbl = document.getElementById('lb-overall-table');
  const sorted = [...ACCURACY].sort((a,b) => (b.overall.accuracy||0) - (a.overall.accuracy||0));

  // Populate month range filters
  const lbMonthFrom = document.getElementById('lb-month-from');
  const lbMonthTo = document.getElementById('lb-month-to');
  MONTHS.forEach((m, i) => {{
    const oFrom = document.createElement('option');
    oFrom.value = m; oFrom.textContent = ml(m);
    lbMonthFrom.appendChild(oFrom);
    const oTo = document.createElement('option');
    oTo.value = m; oTo.textContent = ml(m);
    lbMonthTo.appendChild(oTo);
  }});
  // Default: first to last
  lbMonthFrom.value = MONTHS[0];
  lbMonthTo.value = MONTHS[MONTHS.length - 1];

  function renderOverallTable(fromMonth, toMonth) {{
    // Group by model+config, aggregate across selected month range
    const fromIdx = MONTHS.indexOf(fromMonth);
    const toIdx = MONTHS.indexOf(toMonth);
    const selectedMonths = MONTHS.slice(Math.min(fromIdx, toIdx), Math.max(fromIdx, toIdx) + 1);
    const mm = {{}};
    const filtered = sorted.filter(r => selectedMonths.includes(r.month));
    filtered.forEach(r => {{
      const key = r.model + '|' + r.reasoning_effort;
      if (!mm[key]) mm[key] = {{ model: r.model, effort: r.reasoning_effort, correct: 0, total: 0, tokenSum: 0, tokenCount: 0, timeSum: 0, timeCount: 0 }};
      mm[key].correct += r.overall.correct || 0;
      mm[key].total += r.overall.total || 0;
      if (r.avg_completion_tokens != null) {{
        mm[key].tokenSum += r.avg_completion_tokens * (r.overall.total || 0);
        mm[key].tokenCount += r.overall.total || 0;
      }}
      if (r.avg_elapsed_seconds != null) {{
        mm[key].timeSum += r.avg_elapsed_seconds * (r.overall.total || 0);
        mm[key].timeCount += r.overall.total || 0;
      }}
    }});
    const entries = Object.values(mm).map(e => ({{ ...e, accuracy: e.total > 0 ? e.correct / e.total : 0, avgTokens: e.tokenCount > 0 ? e.tokenSum / e.tokenCount : null, avgTime: e.timeCount > 0 ? e.timeSum / e.timeCount : null }}));
    entries.push({{ model: 'random', effort: '-', correct: 0, total: 0, accuracy: 0.2, avgTokens: null, avgTime: null }});
    entries.sort((a,b) => b.accuracy - a.accuracy);

    let html = '<thead><tr><th>#</th><th></th><th>Model</th><th>Reasoning</th><th>Accuracy</th><th></th><th>Output Tokens / Task</th></tr></thead><tbody>';
    const medals = [String.fromCodePoint(0x1F947), String.fromCodePoint(0x1F948), String.fromCodePoint(0x1F949)];
    entries.forEach((r, i) => {{
      const isRandom = r.model === 'random';
      const pct = (r.accuracy * 100).toFixed(1);
      const rank = i < 3 ? medals[i] : (i+1);
      const avgTok = r.avgTokens != null ? Math.round(r.avgTokens).toLocaleString() : '-';
      const rowStyle = isRandom ? ' style="background:#f1f5f9;color:#64748b;"' : '';
      html += `<tr${{rowStyle}}>
        <td>${{rank}}</td>
        <td style="text-align:center">${{providerLogo(r.model)}}</td>
        <td style="font-weight:600">${{displayModel(r.model)}}</td>
        <td>${{r.effort || '-'}}</td>
        <td class="num"><span class="accuracy-value ${{pctClass(r.accuracy)}}">${{pct}}%</span></td>
        <td><div class="accuracy-bar-wrap"><div class="accuracy-bar" style="width:${{Math.max(pct, 1)}}%"></div></div></td>
        <td class="num">${{avgTok}}</td>
      </tr>`;
    }});
    html += '</tbody>';
    tbl.innerHTML = html;
  }}

  renderOverallTable(MONTHS[0], MONTHS[MONTHS.length - 1]);
  function onMonthRangeChange() {{
    // Ensure From <= To
    const fi = MONTHS.indexOf(lbMonthFrom.value);
    const ti = MONTHS.indexOf(lbMonthTo.value);
    if (fi > ti) lbMonthTo.value = lbMonthFrom.value;
    renderOverallTable(lbMonthFrom.value, lbMonthTo.value);
  }}
  lbMonthFrom.addEventListener('change', onMonthRangeChange);
  lbMonthTo.addEventListener('change', onMonthRangeChange);

  // Accuracy by Month table
  (function() {{
    const tblM = document.getElementById('lb-monthly-table');
    const mFrom = document.getElementById('lb-monthly-from');
    const mTo = document.getElementById('lb-monthly-to');
    MONTHS.forEach(m => {{
      const o1 = document.createElement('option');
      o1.value = m; o1.textContent = ml(m);
      mFrom.appendChild(o1);
      const o2 = document.createElement('option');
      o2.value = m; o2.textContent = ml(m);
      mTo.appendChild(o2);
    }});
    mFrom.value = MONTHS[0];
    mTo.value = MONTHS[MONTHS.length - 1];

    function renderMonthlyTable() {{
      const fi = MONTHS.indexOf(mFrom.value);
      const ti = MONTHS.indexOf(mTo.value);
      const selectedMonths = MONTHS.slice(Math.min(fi, ti), Math.max(fi, ti) + 1);

      // Group by model+config, collect per-month accuracy
      const mm = {{}};
      sorted.forEach(r => {{
        if (!selectedMonths.includes(r.month)) return;
        const key = r.model + '|' + r.reasoning_effort;
        if (!mm[key]) mm[key] = {{ model: r.model, effort: r.reasoning_effort, months: {{}}, totalCorrect: 0, totalAll: 0 }};
        if (!mm[key].months[r.month]) mm[key].months[r.month] = {{ correct: 0, total: 0 }};
        mm[key].months[r.month].correct += r.overall.correct || 0;
        mm[key].months[r.month].total += r.overall.total || 0;
        mm[key].totalCorrect += r.overall.correct || 0;
        mm[key].totalAll += r.overall.total || 0;
      }});
      const entries = Object.values(mm).map(e => ({{ ...e, accuracy: e.totalAll > 0 ? e.totalCorrect / e.totalAll : 0 }}));
      const randomMonths = {{}};
      selectedMonths.forEach(m => {{ randomMonths[m] = {{ correct: 20, total: 100 }}; }});
      entries.push({{ model: 'random', effort: '-', months: randomMonths, totalCorrect: 0, totalAll: 0, accuracy: 0.2 }});
      entries.sort((a,b) => b.accuracy - a.accuracy);

      let html = '<thead><tr><th>#</th><th></th><th>Model</th><th>Reasoning</th>';
      selectedMonths.forEach(m => {{ html += `<th>${{ml(m)}}</th>`; }});
      if (selectedMonths.length > 1) html += '<th>Overall</th>';
      html += '</tr></thead><tbody>';
      const medals = [String.fromCodePoint(0x1F947), String.fromCodePoint(0x1F948), String.fromCodePoint(0x1F949)];
      entries.forEach((e, i) => {{
        const isRandom = e.model === 'random';
        const rank = i < 3 ? medals[i] : (i+1);
        const overallPct = (e.accuracy * 100).toFixed(1);
        const rowStyle = isRandom ? ' style="background:#f1f5f9;color:#64748b;"' : '';
        html += `<tr${{rowStyle}}><td>${{rank}}</td><td style="text-align:center">${{providerLogo(e.model)}}</td><td style="font-weight:600">${{displayModel(e.model)}}</td><td>${{e.effort || '-'}}</td>`;
        selectedMonths.forEach(m => {{
          const d = e.months[m];
          if (d && d.total > 0) {{
            const pct = (d.correct / d.total * 100).toFixed(1);
            html += `<td class="num"><span class="accuracy-value ${{pctClass(d.correct / d.total)}}">${{pct}}%</span></td>`;
          }} else {{
            html += '<td class="num">-</td>';
          }}
        }});
        if (selectedMonths.length > 1) html += `<td class="num"><span class="accuracy-value ${{pctClass(e.accuracy)}}">${{overallPct}}%</span></td>`;
        html += '</tr>';
      }});
      html += '</tbody>';
      tblM.innerHTML = html;
    }}

    renderMonthlyTable();
    function onMonthlyRangeChange() {{
      const fi = MONTHS.indexOf(mFrom.value);
      const ti = MONTHS.indexOf(mTo.value);
      if (fi > ti) mTo.value = mFrom.value;
      renderMonthlyTable();
    }}
    mFrom.addEventListener('change', onMonthlyRangeChange);
    mTo.addEventListener('change', onMonthlyRangeChange);
  }})();

  // Sketch comparison bar chart
  (function() {{
    const sketchCanvas = document.getElementById('chart-sketch');
    const sketchCard = sketchCanvas.closest('.leaderboard-card');
    const sketchEmpty = document.getElementById('sketch-empty');
    const sFrom = document.getElementById('sketch-month-from');
    const sTo = document.getElementById('sketch-month-to');
    if (!SKETCH_ACCURACY || SKETCH_ACCURACY.length === 0) {{
      sketchCard.style.display = 'none';
      return;
    }}
    MONTHS.forEach(m => {{
      const o1 = document.createElement('option');
      o1.value = m; o1.textContent = ml(m);
      sFrom.appendChild(o1);
      const o2 = document.createElement('option');
      o2.value = m; o2.textContent = ml(m);
      sTo.appendChild(o2);
    }});
    sFrom.value = MONTHS[0];
    sTo.value = MONTHS[MONTHS.length - 1];

    let sketchChart = null;

    function renderSketchChart() {{
      const fi = MONTHS.indexOf(sFrom.value);
      const ti = MONTHS.indexOf(sTo.value);
      const selectedMonths = MONTHS.slice(Math.min(fi, ti), Math.max(fi, ti) + 1);
      const sketchMap = {{}};
      SKETCH_ACCURACY.filter(r => selectedMonths.includes(r.month)).forEach(r => {{
        const key = r.model + '|' + r.reasoning_effort;
        if (!sketchMap[key]) sketchMap[key] = {{ model: r.model, effort: r.reasoning_effort, correct: 0, total: 0 }};
        sketchMap[key].correct += r.overall.correct || 0;
        sketchMap[key].total += r.overall.total || 0;
      }});
      const baseMap = {{}};
      sorted.filter(r => selectedMonths.includes(r.month)).forEach(r => {{
        const key = r.model + '|' + r.reasoning_effort;
        if (!baseMap[key]) baseMap[key] = {{ model: r.model, effort: r.reasoning_effort, correct: 0, total: 0 }};
        baseMap[key].correct += r.overall.correct || 0;
        baseMap[key].total += r.overall.total || 0;
      }});
      const rows = [];
      Object.keys(sketchMap).forEach(key => {{
        const s = sketchMap[key];
        const b = baseMap[key];
        if (!b) return;
        const sAcc = s.total > 0 ? (s.correct / s.total) * 100 : 0;
        const bAcc = b.total > 0 ? (b.correct / b.total) * 100 : 0;
        rows.push({{ model: s.model, effort: s.effort, baseAcc: bAcc, sketchAcc: sAcc }});
      }});
      rows.sort((a, b) => b.sketchAcc - a.sketchAcc);

      if (sketchChart) sketchChart.destroy();
      if (rows.length === 0) {{
        sketchCanvas.style.display = 'none';
        sketchEmpty.style.display = 'block';
        return;
      }}
      sketchCanvas.style.display = 'block';
      sketchEmpty.style.display = 'none';

      const labels = rows.map(r => {{
        let name = displayModel(r.model);
        if (r.effort && r.effort !== '-') name += ' (' + r.effort + ')';
        return name;
      }});

      sketchChart = new Chart(sketchCanvas, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{
              label: 'Without Sketch',
              data: rows.map(r => r.baseAcc),
              backgroundColor: 'rgba(59,130,246,0.7)',
              borderColor: '#3b82f6',
              borderWidth: 1,
              borderRadius: 4,
            }},
            {{
              label: 'With Sketch',
              data: rows.map(r => r.sketchAcc),
              backgroundColor: 'rgba(16,185,129,0.7)',
              borderColor: '#10b981',
              borderWidth: 1,
              borderRadius: 4,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            y: {{
              beginAtZero: true,
              max: 100,
              ticks: {{ callback: v => v + '%', font: {{ size: 13 }} }},
              title: {{ display: true, text: 'Accuracy (%)', font: {{ size: 14 }} }}
            }},
            x: {{
              ticks: {{ font: {{ size: 13 }} }}
            }}
          }},
          plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, pointStyle: 'rectRounded', boxWidth: 12, font: {{ size: 14 }}, padding: 16, color: '#1e293b' }} }},
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + '%';
                }}
              }}
            }}
          }}
        }}
      }});
    }}

    renderSketchChart();
    function onSketchRangeChange() {{
      const fi = MONTHS.indexOf(sFrom.value);
      const ti = MONTHS.indexOf(sTo.value);
      if (fi > ti) sTo.value = sFrom.value;
      renderSketchChart();
    }}
    sFrom.addEventListener('change', onSketchRangeChange);
    sTo.addEventListener('change', onSketchRangeChange);
  }})();

  // Group for radar chart
  const modelMap = {{}};
  sorted.forEach(r => {{
    const key = r.model + '|' + r.reasoning_effort;
    if (!modelMap[key]) modelMap[key] = {{ model: r.model, effort: r.reasoning_effort, months: {{}} }};
    modelMap[key].months[r.month] = r.overall;
  }});

  // Model accuracy radar chart (per-category)
  if (sorted.length > 0) {{
    const modelColors = [
      {{bg: 'rgba(59,130,246,0.15)', border: '#3b82f6'}},
      {{bg: 'rgba(239,68,68,0.15)', border: '#ef4444'}},
      {{bg: 'rgba(16,185,129,0.15)', border: '#10b981'}},
      {{bg: 'rgba(245,158,11,0.15)', border: '#f59e0b'}},
      {{bg: 'rgba(139,92,246,0.15)', border: '#8b5cf6'}},
      {{bg: 'rgba(236,72,153,0.15)', border: '#ec4899'}},
      {{bg: 'rgba(99,102,241,0.15)', border: '#6366f1'}},
      {{bg: 'rgba(20,184,166,0.15)', border: '#14b8a6'}},
    ];

    // Populate month filter dropdown
    const radarFilter = document.getElementById('radar-month-filter');
    MONTHS.forEach(m => {{
      const o = document.createElement('option');
      o.value = m; o.textContent = ml(m);
      radarFilter.appendChild(o);
    }});

    let radarChart = null;

    function buildRadarData(filterMonth) {{
      // Group by model+config, aggregate category accuracy
      const radarMap = {{}};
      const filtered = filterMonth ? sorted.filter(r => r.month === filterMonth) : sorted;
      filtered.forEach(r => {{
        const key = r.model + '|' + r.reasoning_effort;
        if (!radarMap[key]) radarMap[key] = {{ model: r.model, effort: r.reasoning_effort, catCorrect: {{}}, catTotal: {{}} }};
        const entry = radarMap[key];
        const ca = r.category_accuracy || {{}};
        CATEGORIES.forEach(c => {{
          if (ca[c]) {{
            entry.catCorrect[c] = (entry.catCorrect[c] || 0) + ca[c].correct;
            entry.catTotal[c] = (entry.catTotal[c] || 0) + ca[c].total;
          }}
        }});
      }});

      return Object.values(radarMap).map((entry, i) => {{
        const col = modelColors[i % modelColors.length];
        return {{
          label: entry.effort ? `${{displayModel(entry.model)}} (${{entry.effort}})` : displayModel(entry.model),
          data: CATEGORIES.map(c => entry.catTotal[c] ? (entry.catCorrect[c] / entry.catTotal[c] * 100) : null),
          backgroundColor: col.bg,
          borderColor: col.border,
          borderWidth: 2,
          pointBackgroundColor: col.border,
          pointRadius: 4,
          fill: true,
        }};
      }});
    }}

    function renderRadar(filterMonth) {{
      const datasets = buildRadarData(filterMonth);
      datasets.unshift({{
        label: 'Random',
        data: CATEGORIES.map(c => c === 'Other' ? null : 20),
        backgroundColor: 'rgba(148,163,184,0.10)',
        borderColor: '#94a3b8',
        borderWidth: 2,
        borderDash: [6, 4],
        pointBackgroundColor: '#94a3b8',
        pointRadius: 3,
        fill: true,
      }});
      if (radarChart) radarChart.destroy();
      radarChart = new Chart(document.getElementById('chart-model-cat'), {{
        type: 'radar',
        data: {{
          labels: CATEGORIES,
          datasets: datasets
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          scales: {{
            r: {{
              beginAtZero: true, max: 100,
              ticks: {{ stepSize: 20, font: {{ size: 14 }}, backdropColor: 'transparent', color: '#1e293b' }},
              pointLabels: {{ font: {{ size: 14 }}, color: '#1e293b' }},
              grid: {{ color: 'rgba(0,0,0,0.1)' }},
              angleLines: {{ color: 'rgba(0,0,0,0.1)' }}
            }}
          }},
          plugins: {{
            legend: {{ position: 'right', labels: {{ usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, font: {{ size: 14 }}, padding: 16, color: '#1e293b' }} }},
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  return ctx.dataset.label + ': ' + (ctx.raw !== null ? ctx.raw.toFixed(1) + '%' : 'N/A');
                }}
              }}
            }}
          }}
        }}
      }});
    }}

    renderRadar('');
    radarFilter.addEventListener('change', () => renderRadar(radarFilter.value));

    // Hide All / Show All toggle
    let radarAllHidden = false;
    document.getElementById('radar-toggle-all').addEventListener('click', function() {{
      if (!radarChart) return;
      radarAllHidden = !radarAllHidden;
      radarChart.data.datasets.forEach((ds, i) => {{
        radarChart.setDatasetVisibility(i, !radarAllHidden);
      }});
      radarChart.update();
      this.textContent = radarAllHidden ? 'Show All' : 'Hide All';
    }});
  }}
}})();
</script>
</body>
</html>"""

    out_path = OUTPUT_DIR / "index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built {out_path} ({len(html_content):,} bytes)")
    print(f"  Months: {MONTHS}")
    print(f"  Examples: {len(examples)}")
    print(f"  Model evals: {len(accuracy)}")


if __name__ == "__main__":
    build()
