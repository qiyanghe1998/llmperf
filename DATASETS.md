# Benchmark Datasets

This document describes the benchmark datasets available for llmperf performance testing.

## Available Datasets

### 1. Wikitext-2
- **Purpose**: Language modeling benchmark
- **Source**: Salesforce/wikitext (Hugging Face)
- **Size**: 4,358 test samples
- **Use Case**: Testing text generation and completion
- **Prompt File**: `datasets/prompts_wikitext.txt` (50 samples)

### 2. MMLU (Massive Multitask Language Understanding)
- **Purpose**: Knowledge and reasoning across 57 academic subjects
- **Source**: cais/mmlu (Hugging Face)
- **Size**: 14,042 test samples
- **Use Case**: Testing factual knowledge and reasoning
- **Prompt File**: `datasets/prompts_mmlu.txt` (50 samples)
- **Format**: Multiple choice questions with A/B/C/D options

### 3. HumanEval
- **Purpose**: Python code generation
- **Source**: openai/openai_humaneval (Hugging Face)
- **Size**: 164 programming problems
- **Use Case**: Testing code generation capabilities
- **Prompt File**: `datasets/prompts_humaneval.txt` (50 samples)
- **Format**: Function signatures with docstrings

## Quick Start

### Download Datasets
```bash
# Install datasets library (if not already installed)
pip install datasets

# Download all benchmark datasets
python download_datasets.py
```

This will create the `datasets/` directory with three prompt files:
- `datasets/prompts_wikitext.txt`
- `datasets/prompts_mmlu.txt`
- `datasets/prompts_humaneval.txt`

### Run Benchmarks

#### Language Modeling (Wikitext)
```bash
python -m src.cli run --model gpt-4o-mini \
  --prompt-file datasets/prompts_wikitext.txt \
  --output results_wikitext.json
```

#### Knowledge & Reasoning (MMLU)
```bash
python -m src.cli run --model gpt-4o-mini \
  --prompt-file datasets/prompts_mmlu.txt \
  --output results_mmlu.json \
  --max-tokens 10
```

#### Code Generation (HumanEval)
```bash
python -m src.cli run --model gpt-4o-mini \
  --prompt-file datasets/prompts_humaneval.txt \
  --output results_humaneval.json \
  --max-tokens 200
```

### Compare Backends

Test the same dataset across different backends:

```bash
# OpenAI
python -m src.cli run --model gpt-4o-mini \
  --prompt-file datasets/prompts_wikitext.txt \
  --output results_openai.json \
  --backend openai

# Ollama (local)
python -m src.cli run --model llama3:latest \
  --prompt-file datasets/prompts_wikitext.txt \
  --output results_ollama.json \
  --backend ollama
```

### Streaming Analysis
```bash
python -m src.cli stream --model gpt-4o-mini \
  --prompt-file datasets/prompts_mmlu.txt \
  --output stream_results.json
```

### Batch Processing
```bash
python -m src.cli batch --model gpt-4o-mini \
  --prompt-file datasets/prompts_humaneval.txt \
  --output batch_results.json \
  --batch-size 4
```

## Dataset Organization

All datasets and prompts are organized in the `datasets/` directory:
```
datasets/
├── prompts_wikitext.txt     # 50 language modeling prompts
├── prompts_mmlu.txt          # 50 knowledge Q&A prompts
└── prompts_humaneval.txt     # 50 code generation prompts
```

## Notes

- The `datasets/` directory is excluded from git by default
- Prompt files contain a subset (50 samples) of each dataset for quick testing
- For full benchmark runs, modify `download_datasets.py` to generate more prompts
- All datasets are cached by Hugging Face after first download

## Customization

To customize the number of prompts or add new datasets, edit `download_datasets.py`:

```python
# Change number of prompts (default: 50)
if len(prompts) >= 100:  # Generate 100 prompts instead
    break
```

## References

- [Wikitext Dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [HumanEval Dataset](https://huggingface.co/datasets/openai/openai_humaneval)

