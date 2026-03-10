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
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("docs")
MONTHS = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
CATEGORIES = [
    "Asymptotic or Limit",
    "Biconditional or Equivalence",
    "Classification or Bijection",
    "Existence",
    "Existential–Universal",
    "General",
    "Implication",
    "Inequality or Bound",
    "Uniqueness",
    "Universal",
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


def collect_stats():
    """Collect per-month, per-category question counts."""
    stats = {}
    for month in MONTHS:
        stats[month] = {}
        for cat in CATEGORIES:
            fpath = DATA_DIR / month / f"qa_{month}_{cat}.json"
            if fpath.exists():
                data = load_json(fpath)
                stats[month][cat] = len(data)
            else:
                stats[month][cat] = 0
    return stats


def collect_examples():
    """Collect a few example questions from each month."""
    examples = []
    seen_ids = set()
    for month in MONTHS:
        for cat in CATEGORIES:
            fpath = DATA_DIR / month / f"qa_{month}_{cat}.json"
            if not fpath.exists():
                continue
            data = load_json(fpath)
            for item in data[:1]:  # take 1 from each category per month
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                examples.append({
                    "id": item["id"],
                    "month": month,
                    "category": cat,
                    "paper_link": item.get("paper_link", ""),
                    "theorem": re.sub(r'\\\\?label\{[^}]*\}\s*', '', item.get("expanded_theorem", item.get("theorem", {}).get("content", ""))).strip(),
                    "sketch": item.get("expanded_sketch", ""),
                    "question": item.get("mcq", {}).get("question", ""),
                    "correct_choice": item.get("mcq", {}).get("correct_choice", {}),
                    "choices": item.get("mcq", {}).get("choices", []),
                })
                if len(examples) >= 40:
                    break
    return examples


def collect_accuracy():
    """Collect model accuracy results from accuracy_test files."""
    results = []
    for fpath in sorted(glob.glob(str(DATA_DIR / "*/accuracy_test_*.json"))):
        # Skip .progress files (incomplete runs)
        if ".progress." in fpath:
            continue
        data = load_json(fpath)
        ti = data.get("test_info", {})
        results.append({
            "file": os.path.basename(fpath),
            "month": ti.get("month", ""),
            "model": ti.get("model", ""),
            "reasoning_effort": ti.get("reasoning_effort", ""),
            "overall": data.get("overall", {}),
            "summary": data.get("summary", {}),
        })
    return results


def build():
    stats = collect_stats()
    examples = collect_examples()
    accuracy = collect_accuracy()

    # JSON-encode data for embedding
    stats_json = json.dumps(stats)
    examples_json = json.dumps(examples)
    accuracy_json = json.dumps(accuracy)
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
    background: linear-gradient(135deg, #1e3a5f 0%, #1a56db 50%, #7c3aed 100%);
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
    background: #f1f5f9;
    font-weight: 600;
    color: var(--text-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  #stats-table th {{
    position: sticky;
    top: 52px;
    z-index: 10;
  }}
  tr:hover td {{ background: #f8fafc; }}
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
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 12px; color: var(--text-muted); }}
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
    background: #f8fafc;
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
    background: #f8fafc;
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
    background: white;
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
    background: #f8fafc;
  }}
  .accuracy-bar {{
    display: inline-block;
    height: 8px;
    border-radius: 4px;
    background: var(--primary);
    min-width: 2px;
    vertical-align: middle;
    margin-left: 8px;
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
    background: white;
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
    <div class="badge">&#x1f4d0; Live Benchmark &mdash; Updated Monthly</div>
    <h1>LiveMathematicianBench</h1>
    <p class="subtitle">
      A live benchmark for evaluating LLMs' capability as mathematicians,
      featuring research-level theorem comprehension from the latest arXiv papers.
    </p>
    <div class="hero-links">
      <a class="btn-primary" href="https://github.com/hendrydong/LiveMathematicianBench" target="_blank" rel="noopener">GitHub</a>
      <a class="btn-primary" href="https://huggingface.co/datasets/hendrydong/bench0303" target="_blank" rel="noopener">Dataset</a>
    </div>
  </div>
</div>

<!-- Nav -->
<div class="nav-tabs">
  <div class="container">
    <div class="nav-tab active" data-tab="leaderboard">Leaderboard</div>
    <div class="nav-tab" data-tab="overview">Overview</div>
    <div class="nav-tab" data-tab="examples">Examples</div>
    <div class="nav-tab" data-tab="about">About</div>
  </div>
</div>

<!-- Sections -->
<div class="container">

  <!-- LEADERBOARD -->
  <div class="section active" id="sec-leaderboard">
    <h2 class="section-title">Model Leaderboard</h2>
    <p class="section-desc">How well do frontier LLMs perform on research-level mathematics? Scores will be updated as more models are evaluated.</p>
    <div id="leaderboard-overall" class="leaderboard-card">
      <h3>Overall Accuracy</h3>
      <div class="table-wrap"><table id="lb-overall-table"></table></div>
    </div>
    <div id="leaderboard-category" class="leaderboard-card">
      <h3>Per-Category Accuracy</h3>
      <div class="table-wrap"><table id="lb-cat-table"></table></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card" style="grid-column: 1 / -1;">
        <h3>Model Accuracy by Category <span style="font-weight:400;font-size:0.8rem;color:var(--text-muted);margin-left:8px;">Click legend to hide/show a model</span></h3>
        <div class="chart-container" style="height:360px;"><canvas id="chart-model-cat"></canvas></div>
      </div>
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
        <h3>Category Breakdown by Month</h3>
        <div class="chart-container" style="height:360px;"><canvas id="chart-cat-monthly"></canvas></div>
      </div>
    </div>
    <h3 style="font-size:1.2rem; margin-top:32px; margin-bottom:16px;">Detailed Statistics</h3>
    <div class="card table-wrap">
      <table id="stats-table">
        <thead><tr><th>Category</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- EXAMPLES -->
  <div class="section" id="sec-examples">
    <h2 class="section-title">Example Questions</h2>
    <p class="section-desc">Browse sample MCQs from the benchmark. Each question is derived from a real arXiv paper theorem.</p>
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
        <p>Each question presents a theorem statement along with a proof sketch, then
           asks the model to identify the correct mathematical conclusion from five
           carefully crafted choices (one correct, one weaker-but-true, and three false).</p>
      </div>
      <div class="card about-card">
        <h3>Theorem Categories</h3>
        <ul>
          <li>Asymptotic or Limit</li>
          <li>Biconditional or Equivalence</li>
          <li>Classification or Bijection</li>
          <li>Existence / Uniqueness</li>
          <li>Implication / Universal</li>
          <li>Inequality or Bound</li>
          <li>General / Existential&ndash;Universal</li>
        </ul>
      </div>
    </div>
  </div>

</div>

<footer>
  <div class="container">
    LiveMathematicianBench &copy; 2025&ndash;2026 &middot;
    <a href="https://github.com/hendrydong/LiveMathematicianBench" target="_blank" rel="noopener">GitHub</a> &middot;
    <a href="https://huggingface.co/datasets/hendrydong/bench0303" target="_blank" rel="noopener">HuggingFace</a>
  </div>
</footer>

<script>
// ===== Embedded Data =====
const MONTHS = {months_json};
const CATEGORIES = {categories_json};
const MONTH_LABELS = {month_labels_json};
const STATS = {stats_json};
const EXAMPLES = {examples_json};
const ACCURACY = {accuracy_json};

const CAT_COLORS = [
  '#3b82f6','#8b5cf6','#06b6d4','#10b981','#f59e0b',
  '#ef4444','#ec4899','#6366f1','#14b8a6','#f97316'
];

function ml(m) {{ return MONTH_LABELS[m] || m; }}
function pctClass(v) {{ return v >= 0.5 ? 'pct-high' : v >= 0.25 ? 'pct-mid' : 'pct-low'; }}

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
  const totalAll = MONTHS.reduce((s,m) => s + CATEGORIES.reduce((a,c) => a + (STATS[m][c]||0), 0), 0);
  const nMonths = MONTHS.length;
  const el = document.getElementById('overview-stats');
  el.innerHTML = `
    <div class="stat-box"><div class="stat-value">${{totalAll}}</div><div class="stat-label">Total Questions</div></div>
    <div class="stat-box"><div class="stat-value">${{nMonths}}</div><div class="stat-label">Monthly Snapshots</div></div>
    <div class="stat-box"><div class="stat-value">${{CATEGORIES.length}}</div><div class="stat-label">Theorem Categories</div></div>
    <div class="stat-box"><div class="stat-value">${{ACCURACY.length > 0 ? ACCURACY.length : '&mdash;'}}</div><div class="stat-label">Model Evaluations</div></div>
  `;
}})();

// ===== Overview Charts =====
(function() {{
  const monthTotals = MONTHS.map(m => CATEGORIES.reduce((a,c) => a + (STATS[m][c]||0), 0));

  new Chart(document.getElementById('chart-monthly'), {{
    type: 'bar',
    data: {{
      labels: MONTHS.map(ml),
      datasets: [{{ label: 'Questions', data: monthTotals, backgroundColor: '#3b82f6', borderRadius: 6 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: true, ticks: {{ precision: 0 }} }} }}
    }}
  }});

  const catTotals = CATEGORIES.map(c => MONTHS.reduce((a,m) => a + (STATS[m][c]||0), 0));
  new Chart(document.getElementById('chart-categories'), {{
    type: 'doughnut',
    data: {{
      labels: CATEGORIES,
      datasets: [{{ data: catTotals, backgroundColor: CAT_COLORS }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }} }}
    }}
  }});

  new Chart(document.getElementById('chart-cat-monthly'), {{
    type: 'bar',
    data: {{
      labels: MONTHS.map(ml),
      datasets: CATEGORIES.map((c,i) => ({{
        label: c, data: MONTHS.map(m => STATS[m][c]||0),
        backgroundColor: CAT_COLORS[i], borderRadius: 3
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }} }} }} }},
      scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, beginAtZero: true, ticks: {{ precision: 0 }} }} }}
    }}
  }});
}})();

// ===== Statistics Table =====
(function() {{
  const thead = document.querySelector('#stats-table thead tr');
  MONTHS.forEach(m => {{ const th = document.createElement('th'); th.textContent = ml(m); th.className = 'num'; thead.appendChild(th); }});
  const thTotal = document.createElement('th'); thTotal.textContent = 'Total'; thTotal.className = 'num'; thead.appendChild(thTotal);

  const tbody = document.querySelector('#stats-table tbody');
  CATEGORIES.forEach(cat => {{
    const tr = document.createElement('tr');
    const tdCat = document.createElement('td'); tdCat.textContent = cat; tr.appendChild(tdCat);
    let rowTotal = 0;
    MONTHS.forEach(m => {{
      const v = STATS[m][cat]||0; rowTotal += v;
      const td = document.createElement('td'); td.textContent = v; td.className = 'num'; tr.appendChild(td);
    }});
    const tdT = document.createElement('td'); tdT.textContent = rowTotal; tdT.className = 'num'; tdT.style.fontWeight = '700'; tr.appendChild(tdT);
    tbody.appendChild(tr);
  }});

  // Total row
  const trT = document.createElement('tr'); trT.style.fontWeight = '700'; trT.style.borderTop = '2px solid var(--border)';
  const tdL = document.createElement('td'); tdL.textContent = 'Total'; trT.appendChild(tdL);
  let grandTotal = 0;
  MONTHS.forEach(m => {{
    const v = CATEGORIES.reduce((a,c) => a + (STATS[m][c]||0), 0); grandTotal += v;
    const td = document.createElement('td'); td.textContent = v; td.className = 'num'; trT.appendChild(td);
  }});
  const tdGT = document.createElement('td'); tdGT.textContent = grandTotal; tdGT.className = 'num'; trT.appendChild(tdGT);
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
            <li class="correct"><span class="choice-label">${{escapeHtml(ex.correct_choice.label)}}</span><span class="math-render" data-field="correct"></span> &#x2705;</li>
            ${{ex.choices.map((ch, ci) => `<li><span class="choice-label">${{escapeHtml(ch.label)}}</span><span class="math-render" data-field="choice-${{ci}}"></span></li>`).join('')}}
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
      setField('correct', ex.correct_choice.text);
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

  // Overall table
  const tbl = document.getElementById('lb-overall-table');
  let html = '<thead><tr><th>#</th><th>Model</th><th>Config</th><th>Month</th><th>Correct</th><th>Total</th><th>Accuracy</th><th></th></tr></thead><tbody>';
  const sorted = [...ACCURACY].sort((a,b) => (b.overall.accuracy||0) - (a.overall.accuracy||0));
  sorted.forEach((r, i) => {{
    const pct = (r.overall.accuracy * 100).toFixed(1);
    html += `<tr>
      <td>${{i+1}}</td>
      <td style="font-weight:600">${{r.model}}</td>
      <td>${{r.reasoning_effort}}</td>
      <td>${{ml(r.month)}}</td>
      <td class="num">${{r.overall.correct}}</td>
      <td class="num">${{r.overall.total}}</td>
      <td class="num"><span class="accuracy-value ${{pctClass(r.overall.accuracy)}}">${{pct}}%</span></td>
      <td><div class="accuracy-bar" style="width:${{Math.max(pct * 1.5, 3)}}px"></div></td>
    </tr>`;
  }});
  html += '</tbody>';
  tbl.innerHTML = html;

  // Per-category table
  const catTbl = document.getElementById('lb-cat-table');
  let catHtml = '<thead><tr><th>Model</th><th>Config</th>';
  CATEGORIES.forEach(c => catHtml += `<th class="num" style="font-size:0.7rem">${{c}}</th>`);
  catHtml += '</tr></thead><tbody>';
  sorted.forEach(r => {{
    catHtml += `<tr><td style="font-weight:600;white-space:nowrap">${{r.model}}</td><td>${{r.reasoning_effort}}</td>`;
    CATEGORIES.forEach(c => {{
      const s = r.summary[c];
      if (s) {{
        const pct = (s.accuracy * 100).toFixed(0);
        catHtml += `<td class="num"><span class="accuracy-value ${{pctClass(s.accuracy)}}">${{pct}}%</span><br><span style="font-size:0.7rem;color:var(--text-muted)">${{s.correct}}/${{s.total}}</span></td>`;
      }} else {{
        catHtml += '<td class="num">&mdash;</td>';
      }}
    }});
    catHtml += '</tr>';
  }});
  catHtml += '</tbody>';
  catTbl.innerHTML = catHtml;

  // Model accuracy chart
  if (sorted.length > 0) {{
    const modelColors = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899'];
    new Chart(document.getElementById('chart-model-cat'), {{
      type: 'radar',
      data: {{
        labels: CATEGORIES.map(c => c.length > 18 ? c.slice(0,16) + '...' : c),
        datasets: sorted.map((r, i) => ({{
          label: `${{r.model}} (${{r.reasoning_effort}})`,
          data: CATEGORIES.map(c => r.summary[c] ? (r.summary[c].accuracy * 100) : 0),
          borderColor: modelColors[i % modelColors.length],
          backgroundColor: modelColors[i % modelColors.length] + '20',
          pointRadius: 3
        }}))
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{ r: {{ beginAtZero: true, max: 100, ticks: {{ stepSize: 20 }} }} }},
        plugins: {{ legend: {{ position: 'bottom' }} }}
      }}
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
