#!/usr/bin/env python3
"""
Analyze benchmark result JSON files and summarize performance metrics.

Features:
- Aggregates latency (mean, p50, p95, p99, min, max), tokens/sec, first-token latency
- Aggregates CPU percent and RSS memory
- Aggregates prompt tokens/sec and context utilization
- Optional MMLU accuracy scoring via either embedded labels in results or a JSONL ground truth

Usage examples:
  python scripts/analyze_results.py --files "results/mmlu5k_qwen_*.json"
  python scripts/analyze_results.py --files "results/mmlu5k_qwen_*.json" \
      --mmlu-jsonl datasets/prompts_mmlu_5k.jsonl --out-csv analysis_summary.csv
"""

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


CHOICE_TO_INDEX = {c: i for i, c in enumerate(["A", "B", "C", "D", "E", "F"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results and optional MMLU accuracy")
    parser.add_argument(
        "--files",
        required=False,
        default="results/*.json",
        help="Glob for result JSON files (default: results/*.json)",
    )
    parser.add_argument(
        "--mmlu-jsonl",
        required=False,
        default=None,
        help="Optional MMLU JSONL with ground truth (id, answer_index). Used if results lack labels.",
    )
    parser.add_argument(
        "--out-csv",
        required=False,
        default=None,
        help="Optional CSV path to write per-file summary rows",
    )
    return parser.parse_args()


def list_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.isfile(f)]


def extract_choice(text: str) -> int:
    if not text:
        return -1
    t = text.strip()
    m = re.search(r"(?i)(answer\s*[:=]\s*|final\s*answer\s*[:=]\s*)\(?([A-F])\)?", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(2).upper(), -1)
    m = re.search(r"\(([A-F])\)\s*$", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(1).upper(), -1)
    m = re.search(r"\b([A-F])\b\s*$", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(1).upper(), -1)
    return -1


def pick_numeric(item: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in item and isinstance(item[k], (int, float)):
            try:
                val = float(item[k])
            except Exception:
                continue
            if math.isfinite(val):
                return val
    return None


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)

    def pct(p: float) -> float:
        if not xs:
            return float("nan")
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs[int(k)]
        d0 = xs[f] * (c - k)
        d1 = xs[c] * (k - f)
        return d0 + d1

    return {
        "mean": sum(xs) / len(xs),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": xs[0],
        "max": xs[-1],
    }


METRIC_KEYS: Dict[str, List[str]] = {
    "total_latency_ms": ["total_latency_ms", "latency_ms", "total_ms", "elapsed_ms"],
    "first_token_latency_ms": ["first_token_latency_ms", "ttft_ms"],
    "tokens_per_sec": ["tokens_per_sec"],
    "cpu_percent": ["cpu_percent"],
    "mem_rss_mb": ["mem_rss_mb"],
    "prompt_tokens_per_sec": ["prompt_tokens_per_sec"],
    "ctx_utilization": ["ctx_utilization"],
}


@dataclass
class FileSummary:
    filename: str
    model: str
    samples: int
    metrics: Dict[str, Dict[str, float]]
    accuracy_pct: Optional[float]


def infer_model_name(filename: str, items: List[Dict[str, Any]]) -> str:
    # Prefer explicit model_name field in items
    for it in items:
        m = it.get("model_name") or it.get("model")
        if isinstance(m, str) and m:
            return m
    # Fallback: derive from filename
    base = os.path.basename(filename)
    m = re.sub(r"\.json$", "", base)
    return m


def load_ground_truth_map(jsonl_path: Optional[str]) -> Dict[str, int]:
    if not jsonl_path:
        return {}
    mapping: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("id"))
            if qid is None:
                continue
            mapping[qid] = int(obj.get("answer_index"))
    return mapping


def compute_accuracy(items: List[Dict[str, Any]], gt_map: Dict[str, int]) -> Tuple[int, int]:
    total = 0
    correct = 0
    for it in items:
        # Prefer embedded label if present
        label_idx: Optional[int] = None
        if "answer_index" in it:
            try:
                label_idx = int(it["answer_index"])  # embedded
            except Exception:
                label_idx = None
        if label_idx is None and gt_map:
            qid = str(it.get("id")) if it.get("id") is not None else None
            if qid is not None:
                label_idx = gt_map.get(qid)
        if label_idx is None:
            continue
        pred_idx = extract_choice(it.get("text", ""))
        total += 1
        if pred_idx == label_idx:
            correct += 1
    return correct, total


def summarize_file(path: str, gt_map: Dict[str, int]) -> FileSummary:
    with open(path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    model_name = infer_model_name(path, items)
    metrics_values: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for it in items:
        for m, keys in METRIC_KEYS.items():
            val = pick_numeric(it, keys)
            if val is not None:
                metrics_values[m].append(val)

    metric_summaries: Dict[str, Dict[str, float]] = {
        m: compute_percentiles(vals) for m, vals in metrics_values.items() if vals
    }

    acc_pct: Optional[float] = None
    if items:
        correct, total = compute_accuracy(items, gt_map)
        if total > 0:
            acc_pct = round(100.0 * correct / total, 2)

    return FileSummary(
        filename=os.path.basename(path),
        model=model_name,
        samples=len(items),
        metrics=metric_summaries,
        accuracy_pct=acc_pct,
    )


def print_human_summary(summaries: List[FileSummary]) -> None:
    print("MODEL, samples, acc%, total_ms.mean, total_ms.p95, total_ms.p99, ttft_ms.mean, tok/s.mean, CPU%.mean, RSS_MB.mean")
    for s in summaries:
        m = s.metrics
        t = m.get("total_latency_ms", {})
        tt = m.get("first_token_latency_ms", {})
        tok = m.get("tokens_per_sec", {})
        cpu = m.get("cpu_percent", {})
        rss = m.get("mem_rss_mb", {})
        print(
            f"{s.model}, {s.samples}, {s.accuracy_pct if s.accuracy_pct is not None else ''}, "
            f"{t.get('mean', float('nan')):.2f}, {t.get('p95', float('nan')):.2f}, {t.get('p99', float('nan')):.2f}, "
            f"{tt.get('mean', float('nan')):.2f}, {tok.get('mean', float('nan')):.2f}, {cpu.get('mean', float('nan')):.2f}, {rss.get('mean', float('nan')):.2f}"
        )

    print("\nExtra: prompt_tok/s.mean, ctx_util.mean")
    for s in summaries:
        pps = s.metrics.get("prompt_tokens_per_sec", {})
        ctx = s.metrics.get("ctx_utilization", {})
        print(f"{s.model}, {pps.get('mean', float('nan')):.2f}, {ctx.get('mean', float('nan')):.2f}")


def write_csv(summaries: List[FileSummary], out_csv: str) -> None:
    import csv
    fields = [
        "filename",
        "model",
        "samples",
        "accuracy_pct",
        # main metric stats
        "total_ms_mean",
        "total_ms_p50",
        "total_ms_p95",
        "total_ms_p99",
        "ttft_ms_mean",
        "tokens_per_sec_mean",
        "cpu_percent_mean",
        "mem_rss_mb_mean",
        "prompt_tokens_per_sec_mean",
        "ctx_utilization_mean",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            t = s.metrics.get("total_latency_ms", {})
            tt = s.metrics.get("first_token_latency_ms", {})
            tok = s.metrics.get("tokens_per_sec", {})
            cpu = s.metrics.get("cpu_percent", {})
            rss = s.metrics.get("mem_rss_mb", {})
            pps = s.metrics.get("prompt_tokens_per_sec", {})
            ctx = s.metrics.get("ctx_utilization", {})
            w.writerow({
                "filename": s.filename,
                "model": s.model,
                "samples": s.samples,
                "accuracy_pct": s.accuracy_pct if s.accuracy_pct is not None else "",
                "total_ms_mean": f"{t.get('mean', float('nan')):.2f}",
                "total_ms_p50": f"{t.get('p50', float('nan')):.2f}",
                "total_ms_p95": f"{t.get('p95', float('nan')):.2f}",
                "total_ms_p99": f"{t.get('p99', float('nan')):.2f}",
                "ttft_ms_mean": f"{tt.get('mean', float('nan')):.2f}",
                "tokens_per_sec_mean": f"{tok.get('mean', float('nan')):.2f}",
                "cpu_percent_mean": f"{cpu.get('mean', float('nan')):.2f}",
                "mem_rss_mb_mean": f"{rss.get('mean', float('nan')):.2f}",
                "prompt_tokens_per_sec_mean": f"{pps.get('mean', float('nan')):.2f}",
                "ctx_utilization_mean": f"{ctx.get('mean', float('nan')):.2f}",
            })


def main() -> None:
    args = parse_args()
    files = list_files(args.files)
    if not files:
        print(f"No files matched: {args.files}")
        return
    gt_map = load_ground_truth_map(args.mmlu_jsonl)

    summaries = [summarize_file(fp, gt_map) for fp in files]
    print_human_summary(summaries)
    if args.out_csv:
        write_csv(summaries, args.out_csv)
        print(f"\nWrote CSV summary to {args.out_csv}")


if __name__ == "__main__":
    main()


