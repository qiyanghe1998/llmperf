#!/usr/bin/env python3
"""Aggregate results from all JSON files and print overall statistics."""

import json
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics

def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all result JSON files from the results directory."""
    pattern = os.path.join(results_dir, "results_*.json")
    files = glob.glob(pattern)
    
    all_results = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    all_results.append(results)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}", file=sys.stderr)
    
    return all_results

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values."""
    if not values:
        return {}
    
    values.sort()
    n = len(values)
    
    return {
        'count': n,
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p95': values[int(n * 0.95)] if n > 0 else 0,
        'p99': values[int(n * 0.99)] if n > 0 else 0,
        'std': statistics.stdev(values) if n > 1 else 0
    }

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results and calculate overall statistics."""
    if not results:
        return {}
    
    # Extract all metrics
    total_latencies = []
    first_token_latencies = []
    tokens_per_sec = []
    cpu_percentages = []
    mem_rss_mb = []
    total_tokens = []
    input_tokens = []
    output_tokens = []
    
    # Group by model
    by_model = {}
    by_dataset = {}
    
    for result in results:
        # Collect metrics
        if result.get('total_latency_ms'):
            total_latencies.append(result['total_latency_ms'])
        if result.get('first_token_latency_ms'):
            first_token_latencies.append(result['first_token_latency_ms'])
        if result.get('tokens_per_sec'):
            tokens_per_sec.append(result['tokens_per_sec'])
        if result.get('cpu_percent'):
            cpu_percentages.append(result['cpu_percent'])
        if result.get('mem_rss_mb'):
            mem_rss_mb.append(result['mem_rss_mb'])
        if result.get('total_tokens'):
            total_tokens.append(result['total_tokens'])
        if result.get('input_tokens'):
            input_tokens.append(result['input_tokens'])
        if result.get('output_tokens'):
            output_tokens.append(result['output_tokens'])
        
        # Group by model
        model = result.get('model_name', 'unknown')
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(result)
        
        # Group by dataset (infer from prompt content or filename)
        prompt = result.get('prompt', '')
        if 'def ' in prompt and '->' in prompt:
            dataset = 'humaneval'
        elif len(prompt) > 100 and '=' not in prompt[:50]:
            dataset = 'wikitext'
        else:
            dataset = 'mmlu'
        
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(result)
    
    # Calculate overall stats
    overall_stats = {
        'total_prompts': len(results),
        'total_latency_ms': calculate_stats(total_latencies),
        'first_token_latency_ms': calculate_stats(first_token_latencies),
        'tokens_per_sec': calculate_stats(tokens_per_sec),
        'cpu_percent': calculate_stats(cpu_percentages),
        'mem_rss_mb': calculate_stats(mem_rss_mb),
        'total_tokens': calculate_stats(total_tokens),
        'input_tokens': calculate_stats(input_tokens),
        'output_tokens': calculate_stats(output_tokens),
    }
    
    # Calculate per-model stats
    model_stats = {}
    for model, model_results in by_model.items():
        model_latencies = [r['total_latency_ms'] for r in model_results if r.get('total_latency_ms')]
        model_tokens_per_sec = [r['tokens_per_sec'] for r in model_results if r.get('tokens_per_sec')]
        
        model_stats[model] = {
            'prompts': len(model_results),
            'total_latency_ms': calculate_stats(model_latencies),
            'tokens_per_sec': calculate_stats(model_tokens_per_sec),
        }
    
    # Calculate per-dataset stats
    dataset_stats = {}
    for dataset, dataset_results in by_dataset.items():
        dataset_latencies = [r['total_latency_ms'] for r in dataset_results if r.get('total_latency_ms')]
        dataset_tokens_per_sec = [r['tokens_per_sec'] for r in dataset_results if r.get('tokens_per_sec')]
        
        dataset_stats[dataset] = {
            'prompts': len(dataset_results),
            'total_latency_ms': calculate_stats(dataset_latencies),
            'tokens_per_sec': calculate_stats(dataset_tokens_per_sec),
        }
    
    return {
        'overall': overall_stats,
        'by_model': model_stats,
        'by_dataset': dataset_stats
    }

def print_stats(stats: Dict[str, Any]):
    """Print formatted statistics."""
    print("=" * 80)
    print("OVERALL BENCHMARK STATISTICS")
    print("=" * 80)
    
    overall = stats.get('overall', {})
    print(f"Total Prompts Processed: {overall.get('total_prompts', 0)}")
    print()
    
    # Overall latency stats
    latency_stats = overall.get('total_latency_ms', {})
    if latency_stats:
        print("TOTAL LATENCY (ms):")
        print(f"  Count:     {latency_stats.get('count', 0)}")
        print(f"  Min:       {latency_stats.get('min', 0):.1f}")
        print(f"  Max:       {latency_stats.get('max', 0):.1f}")
        print(f"  Mean:      {latency_stats.get('mean', 0):.1f}")
        print(f"  Median:    {latency_stats.get('median', 0):.1f}")
        print(f"  P95:       {latency_stats.get('p95', 0):.1f}")
        print(f"  P99:       {latency_stats.get('p99', 0):.1f}")
        print(f"  Std Dev:   {latency_stats.get('std', 0):.1f}")
        print()
    
    # Overall tokens/sec stats
    tps_stats = overall.get('tokens_per_sec', {})
    if tps_stats:
        print("TOKENS PER SECOND:")
        print(f"  Count:     {tps_stats.get('count', 0)}")
        print(f"  Min:       {tps_stats.get('min', 0):.2f}")
        print(f"  Max:       {tps_stats.get('max', 0):.2f}")
        print(f"  Mean:      {tps_stats.get('mean', 0):.2f}")
        print(f"  Median:    {tps_stats.get('median', 0):.2f}")
        print(f"  P95:       {tps_stats.get('p95', 0):.2f}")
        print(f"  P99:       {tps_stats.get('p99', 0):.2f}")
        print()
    
    # First token latency
    first_token_stats = overall.get('first_token_latency_ms', {})
    if first_token_stats:
        print("FIRST TOKEN LATENCY (ms):")
        print(f"  Count:     {first_token_stats.get('count', 0)}")
        print(f"  Min:       {first_token_stats.get('min', 0):.1f}")
        print(f"  Max:       {first_token_stats.get('max', 0):.1f}")
        print(f"  Mean:      {first_token_stats.get('mean', 0):.1f}")
        print(f"  Median:    {first_token_stats.get('median', 0):.1f}")
        print(f"  P95:       {first_token_stats.get('p95', 0):.1f}")
        print(f"  P99:       {first_token_stats.get('p99', 0):.1f}")
        print()
    
    # Per-model breakdown
    print("PER-MODEL BREAKDOWN:")
    print("-" * 40)
    for model, model_stats in stats.get('by_model', {}).items():
        print(f"Model: {model}")
        print(f"  Prompts: {model_stats.get('prompts', 0)}")
        
        model_latency = model_stats.get('total_latency_ms', {})
        if model_latency:
            print(f"  Avg Latency: {model_latency.get('mean', 0):.1f}ms")
            print(f"  P95 Latency: {model_latency.get('p95', 0):.1f}ms")
        
        model_tps = model_stats.get('tokens_per_sec', {})
        if model_tps:
            print(f"  Avg Tokens/sec: {model_tps.get('mean', 0):.2f}")
        print()
    
    # Per-dataset breakdown
    print("PER-DATASET BREAKDOWN:")
    print("-" * 40)
    for dataset, dataset_stats in stats.get('by_dataset', {}).items():
        print(f"Dataset: {dataset.upper()}")
        print(f"  Prompts: {dataset_stats.get('prompts', 0)}")
        
        dataset_latency = dataset_stats.get('total_latency_ms', {})
        if dataset_latency:
            print(f"  Avg Latency: {dataset_latency.get('mean', 0):.1f}ms")
            print(f"  P95 Latency: {dataset_latency.get('p95', 0):.1f}ms")
        
        dataset_tps = dataset_stats.get('tokens_per_sec', {})
        if dataset_tps:
            print(f"  Avg Tokens/sec: {dataset_tps.get('mean', 0):.2f}")
        print()
    
    print("=" * 80)

def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} does not exist")
        sys.exit(1)
    
    # Load and aggregate results
    results = load_all_results(results_dir)
    if not results:
        print("No results found in the directory")
        sys.exit(1)
    
    stats = aggregate_results(results)
    print_stats(stats)

if __name__ == "__main__":
    main()
