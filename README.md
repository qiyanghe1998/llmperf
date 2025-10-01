# llmperf  
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![PyPI version](https://img.shields.io/badge/pypi-vllmperf-blue.svg)](https://pypi.org/project/llmperf)

A lightweight CLI toolkit for benchmarking and profiling large language model (LLM) inference across multiple backends. llmperf makes it easy to measure per-token latency, cost, batching vs. single-call trade-offs, and streaming performance, producing detailed CSV/JSON reports for analysis.

---

## Table of Contents
- [Features](#features)  
- [Installation](#installation)  
- [Quickstart](#quickstart)  
- [Usage](#usage)  
- [Development](#development)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features
- **Multi-model profiling**: Compare hosted (OpenAI) and local (vLLM, Ollama) inference.  
- **Token-level metrics**: Capture latency, token counts, and cost per prompt.  
- **Batching analysis**: Evaluate throughput vs. latency across different batch sizes.  
- **Streaming support**: Measure token-by-token latency and tail-latency buckets.  
- **Prompt mutation**: Automatically generate and compare prompt variants.  
- **Plugin-friendly**: Add new inference backends via a simple plugin API.  
- **Output formats**: Export results to CSV or JSON for downstream analysis or dashboards.  

---

## Installation

```bash
# Clone
git clone https://github.com/<your-org>/llmperf.git
cd llmperf
```

### Prerequisites

- Python 3.10+ (recommended 3.12)
- Use a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### Install (macOS, CPU)

vLLM is Linux/GPU-first and can fail to install on macOS via pip. On macOS install core deps only:
```
pip install -r requirements-macos.txt
```

### Install (Linux with NVIDIA GPU)

Install PyTorch that matches your CUDA, then vLLM and core deps. Example for CUDA 12.1:
```
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2
pip install 'vllm>=0.7,<0.11'
pip install -r requirements-core.txt
```

### Docker (recommended on macOS for vLLM)

If you need vLLM on a Mac, use Docker with GPU on a Linux host, or run CPU-only inside a container. Example GPU container (Linux with NVIDIA drivers + nvidia-container-toolkit installed):
```
docker run --gpus all -it --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:24.02-py3 \
  bash -lc "pip install 'vllm>=0.7,<0.11' && pip install -r requirements-core.txt"
```

### From PyPI (once published)
```
pip install llmperf
```

## Quickstart

1. **Setup configuration** (recommended):
    ```bash
    # Copy example config
    cp config.env.example config.env
    
    # Edit config.env with your API key
    # OPENAI_API_KEY=sk-your-actual-key-here
    ```

2. **Prepare prompts**:
    ```bash
    echo "Hello, world!" > prompts.txt
    echo "What is the capital of France?" >> prompts.txt
    ```

3. **Run experiments**:
    ```bash
    # Using config file (API key loaded automatically)
    python -m src.cli run --model gpt-4o-mini --prompt-file prompts.txt --output results.json
    
    # Or with explicit API key
    python -m src.cli run --model gpt-4o-mini --prompt-file prompts.txt --output results.json --api-key sk-your-key
    ```

4. **Inspect results**: Check `results.json` for token-level latency and output text.

### Platform-Specific Examples

**macOS (OpenAI + Ollama)**:
```bash
# OpenAI (requires API key in config.env)
python -m src.cli run --model gpt-4o-mini --prompt-file prompts.txt --output results.json

# Ollama (local, no API key needed)
python -m src.cli run --model llama3:latest --prompt-file prompts.txt --output results.json --backend ollama
```

**Linux (All backends)**:
```bash
# vLLM (local server)
python -m src.cli run --model http://localhost:8000/v1 --prompt-file prompts.txt --output results.json --backend vllm
```

Optional (Linux/GPU only): run against a local vLLM server. Start vLLM separately, then:
```
llmperf run --model http://localhost:8000/v1 --prompt-file prompts.txt --output results.json
```

## Usage

```
llmperf [COMMAND] [OPTIONS]
```
| Command   | Description                                                  |
| --------- | ------------------------------------------------------------ |
| `run`     | Single-call inference: measure end-to-end latency and output |
| `profile` | Token-level profiling: capture per-token latencies           |
| `batch`   | Batch inference: test throughput vs. latency trade-offs      |
| `stream`  | Streaming mode: measure token-streaming latency buckets      |

Common options:

- --model TEXT (e.g. gpt-2, gpt-4, mistral-7b)

- --prompt-file PATH (one prompt per line)

- --batch-size INT (for batch)

- --output PATH (.json or .csv)

Run llmperf COMMAND --help for full option lists and examples.

## Configuration

llmperf supports configuration files for easier API key management:

### Setup
```bash
# Copy example configuration
cp config.env.example config.env

# Edit config.env with your settings
# OPENAI_API_KEY=sk-your-actual-key-here
# OPENAI_BASE_URL=https://api.openai.com/v1  # optional
```

### Usage
- **Config file**: API keys loaded automatically from `config.env`
- **Command-line override**: `--api-key` and `--base-url` override config file
- **Environment variables**: Still supported as fallback

### Security
- `config.env` is excluded from git by default
- Never commit API keys to version control
- Use `config.env.example` as a template

## Development

Set up a development environment and run tests/lint:
```
# Install dev dependencies (core only; see platform notes above)
pip install -r requirements-core.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linter
flake8
```

To add a new backend plugin, create a Python module under `llmperf/plugins/` implementing the standard interface. See `llmperf/plugins/example.py` for a template.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and create a feature branch.

2. Discuss major changes by opening an issue.

3. Implement your feature and add tests.

4. Submit a pull request for review.

We appreciate your help in making llmperf better!

## License

This project is licensed under the MIT License. See [LICENSE](https://img.shields.io/badge/license-MIT-blue.svg) for details.

