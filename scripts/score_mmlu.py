#!/usr/bin/env python3
"""Score MMLU accuracy from JSON results using MMLU JSONL ground truth.

Usage:
  python scripts/score_mmlu.py --results-dir results --mmlu-jsonl datasets/prompts_mmlu.jsonl --output scores_mmlu.csv
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

CHOICE_TO_INDEX = {c: i for i, c in enumerate(['A', 'B', 'C', 'D', 'E', 'F'])}

def load_ground_truth(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    gt: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            qid = obj['id']
            gt[qid] = obj
    return gt

def list_result_files(results_dir: str) -> List[str]:
    # Support both legacy and new naming schemes
    patterns = [
        os.path.join(results_dir, 'results_prompts_mmlu_*.json'),
        os.path.join(results_dir, 'mmlu*_qwen_*.json'),
        os.path.join(results_dir, 'mmlu*_*.json'),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def extract_choice(text: str) -> int:
    """Extract chosen option index from model output text.

    Heuristics:
    - Look for a single capital letter A-F as the final answer
    - Look for patterns like 'Answer: C' or '(C)'
    """
    if not text:
        return -1
    t = text.strip()
    # Common strict formats: "Final Answer: C", "Answer: D"
    m = re.search(r"(?i)(answer\s*[:=]\s*|final\s*answer\s*[:=]\s*)\(?([A-F])\)?", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(2).upper(), -1)
    # Letter in parentheses at end: "...(C)"
    m = re.search(r"\(([A-F])\)\s*$", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(1).upper(), -1)
    # Lone capital letter at end of text
    m = re.search(r"\b([A-F])\b\s*\.?\s*$", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(1).upper(), -1)
    # Phrases: "The answer is C", "Option C", "Choice C"
    m = re.search(r"(?i)(the\s+answer\s+is|option|choice)\s*([A-F])\b", t)
    if m:
        return CHOICE_TO_INDEX.get(m.group(2).upper(), -1)
    return -1

def score_results(files: List[str], ground_truth: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            batch = json.load(f)
        for item in batch:
            qid = item.get('id')
            subject = item.get('subject')
            model = item.get('model_name')
            text = item.get('text')
            pred_idx = extract_choice(text)
            gt_obj = ground_truth.get(qid)
            if gt_obj is None:
                continue
            label_idx = int(gt_obj['answer_index'])
            correct = int(pred_idx == label_idx)
            rows.append({
                'id': qid,
                'subject': subject,
                'model': model,
                'pred_index': pred_idx,
                'label_index': label_idx,
                'correct': correct
            })
    return rows

def aggregate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_model: Dict[str, List[int]] = {}
    by_model_subject: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        m = r['model']
        s = r['subject'] or 'unknown'
        by_model.setdefault(m, []).append(r['correct'])
        by_model_subject.setdefault((m, s), []).append(r['correct'])
    per_model = []
    for m, vals in by_model.items():
        total = len(vals)
        acc = sum(vals) / total if total else 0.0
        per_model.append({'model': m, 'samples': total, 'accuracy': round(acc * 100, 2)})
    per_model_subject = []
    for (m, s), vals in by_model_subject.items():
        total = len(vals)
        acc = sum(vals) / total if total else 0.0
        per_model_subject.append({'model': m, 'subject': s, 'samples': total, 'accuracy': round(acc * 100, 2)})
    per_model.sort(key=lambda x: x['model'])
    per_model_subject.sort(key=lambda x: (x['model'], x['subject']))
    return per_model, per_model_subject

def write_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        return
    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', help='Directory with results_*.json files')
    parser.add_argument('--mmlu-jsonl', default='datasets/prompts_mmlu.jsonl', help='Ground truth JSONL with ids/answers')
    parser.add_argument('--output', default='scores_mmlu.csv', help='Output CSV path')
    parser.add_argument('--output-by-subject', default='scores_mmlu_by_subject.csv', help='Output CSV per subject')
    args = parser.parse_args()

    gt = load_ground_truth(args.mmlu_jsonl)
    files = list_result_files(args.results_dir)
    rows = score_results(files, gt)
    per_model, per_model_subject = aggregate(rows)

    write_csv(per_model, args.output)
    write_csv(per_model_subject, args.output_by_subject)

    print(f"Wrote {args.output} and {args.output_by_subject}")

if __name__ == '__main__':
    main()


