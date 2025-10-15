#!/usr/bin/env python3
"""
Download benchmark datasets for LLM performance testing.

This script downloads three popular datasets:
1. Wikitext - Language modeling benchmark
2. MMLU - Massive Multitask Language Understanding
3. HumanEval - Code generation benchmark

Usage:
    python download_datasets.py
"""

from datasets import load_dataset
import json
import os


def download_wikitext():
    """Download Wikitext dataset."""
    print("=" * 60)
    print("Downloading Wikitext-2...")
    print("=" * 60)
    
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs('datasets', exist_ok=True)
        
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        print(f"✓ Wikitext-2 loaded: {len(dataset)} samples")
        
        # Save all non-empty samples as prompts
        prompts = []
        for item in dataset:
            text = item['text'].strip()
            if text:
                prompts.append(text)
        
        with open('datasets/prompts_wikitext.txt', 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        
        print(f"✓ Saved {len(prompts)} prompts to datasets/prompts_wikitext.txt")
        return dataset
    
    except Exception as e:
        print(f"✗ Error downloading Wikitext: {e}")
        return None


def download_mmlu():
    """Download MMLU dataset."""
    print("\n" + "=" * 60)
    print("Downloading MMLU (Massive Multitask Language Understanding)...")
    print("=" * 60)
    
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs('datasets', exist_ok=True)
        
        dataset = load_dataset('cais/mmlu', 'all', split='test')
        print(f"✓ MMLU loaded: {len(dataset)} samples")
        
        # Create prompts from all questions
        prompts: list[str] = []
        mmlu_jsonl_path = 'datasets/prompts_mmlu.jsonl'
        with open(mmlu_jsonl_path, 'w', encoding='utf-8') as jf:
            for i, item in enumerate(dataset):
                question = item['question']
                choices = item['choices']
                answer = item['answer']  # expected index into choices (0-based)
                subject = item.get('subject', 'unknown')

                # Format text prompt (for text-mode compatibility)
                prompt = f"{question}\n"
                for idx, choice in enumerate(choices):
                    prompt += f"{chr(65+idx)}) {choice}\n"
                prompts.append(prompt.strip())

                # Write JSONL with id and metadata
                record = {
                    'id': f'mmlu-{i}',
                    'subject': subject,
                    'question': question,
                    'choices': choices,
                    'answer_index': int(answer)
                }
                import json as _json
                jf.write(_json.dumps(record, ensure_ascii=False) + '\n')

        with open('datasets/prompts_mmlu.txt', 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')

        print(f"✓ Saved {len(prompts)} prompts to datasets/prompts_mmlu.txt")
        print(f"✓ Saved JSONL with ids/answers to {mmlu_jsonl_path}")
        return dataset
    
    except Exception as e:
        print(f"✗ Error downloading MMLU: {e}")
        return None


def download_humaneval():
    """Download HumanEval dataset."""
    print("\n" + "=" * 60)
    print("Downloading HumanEval (Code Generation)...")
    print("=" * 60)
    
    try:
        # Create datasets directory if it doesn't exist
        os.makedirs('datasets', exist_ok=True)
        
        dataset = load_dataset('openai/openai_humaneval', split='test')
        print(f"✓ HumanEval loaded: {len(dataset)} samples")
        
        # Create prompts from all problems
        prompts = []
        for item in dataset:
            prompt = item['prompt']
            prompts.append(prompt.strip())
        
        with open('datasets/prompts_humaneval.txt', 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        
        print(f"✓ Saved {len(prompts)} prompts to datasets/prompts_humaneval.txt")
        return dataset
    
    except Exception as e:
        print(f"✗ Error downloading HumanEval: {e}")
        return None


def main():
    """Main function to download all datasets."""
    print("\n🚀 Starting dataset download...\n")
    
    # Download datasets
    wikitext = download_wikitext()
    mmlu = download_mmlu()
    humaneval = download_humaneval()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    print(f"Wikitext:   {'✓ Success' if wikitext else '✗ Failed'}")
    print(f"MMLU:       {'✓ Success' if mmlu else '✗ Failed'}")
    print(f"HumanEval:  {'✓ Success' if humaneval else '✗ Failed'}")
    
    print("\n📁 Generated Prompt Files:")
    if wikitext:
        print("  - datasets/prompts_wikitext.txt (language modeling)")
    if mmlu:
        print("  - datasets/prompts_mmlu.txt (knowledge & reasoning)")
    if humaneval:
        print("  - datasets/prompts_humaneval.txt (code generation)")
    
    print("\n✨ Usage examples:")
    print("  python -m src.cli run --model gpt-4o-mini --prompt-file datasets/prompts_wikitext.txt --output wikitext_results.json")
    print("  python -m src.cli run --model gpt-4o-mini --prompt-file datasets/prompts_mmlu.txt --output mmlu_results.json")
    print("  python -m src.cli run --model gpt-4o-mini --prompt-file datasets/prompts_humaneval.txt --output humaneval_results.json")
    print()


if __name__ == "__main__":
    main()

