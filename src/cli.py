"""Command-line interface for llmperf."""

import asyncio
import click
import json
import csv
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .plugins import BaseLLMBackend
from .plugins.openai import OpenAIBackend
from .plugins.example import ExampleBackend

# Import other backends only when needed
import sys
if sys.platform != 'darwin':
    from .plugins.vllm import VLLMBackend

# Import Ollama only if aiohttp is available
try:
    from .plugins.ollama import OllamaBackend
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from a text file, one per line."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_config(config_file: str = "config.env") -> Dict[str, str]:
    """Load configuration from a file."""
    config = {}
    config_path = Path(config_file)
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    return config


def save_results_json(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_results_csv(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV file."""
    if not results:
        return
    
    fieldnames = results[0].keys()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def get_backend(model_name: str, backend_type: str = "auto", **kwargs) -> BaseLLMBackend:
    """Get the appropriate backend based on model name and type."""
    if backend_type == "openai" or model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return OpenAIBackend(model_name, **kwargs)
    elif backend_type == "vllm" or model_name.startswith("http://") or model_name.startswith("https://"):
        if sys.platform != 'darwin':
            return VLLMBackend(model_name, **kwargs)
        else:
            raise ValueError("vLLM backend is not supported on macOS")
    elif backend_type == "ollama":
        if OLLAMA_AVAILABLE:
            return OllamaBackend(model_name, **kwargs)
        else:
            raise ValueError("Ollama backend requires aiohttp to be installed")
    elif backend_type == "example":
        return ExampleBackend(model_name, **kwargs)
    else:
        # Auto-detect based on model name
        if model_name.startswith("gpt-") or model_name.startswith("o1-"):
            return OpenAIBackend(model_name, **kwargs)
        elif model_name.startswith("http://") or model_name.startswith("https://"):
            if sys.platform != 'darwin':
                return VLLMBackend(model_name, **kwargs)
            else:
                raise ValueError("vLLM backend is not supported on macOS")
        else:
            # Default to Ollama for local models if available, otherwise example
            if OLLAMA_AVAILABLE:
                return OllamaBackend(model_name, **kwargs)
            else:
                return ExampleBackend(model_name, **kwargs)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """llmperf - A lightweight CLI toolkit for benchmarking and profiling LLM inference."""
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='Model name (e.g., gpt-4o-mini, mistral-7b)')
@click.option('--prompt-file', '-f', required=True, help='File containing prompts (one per line)')
@click.option('--output', '-o', required=True, help='Output file (.json or .csv)')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, help='Sampling temperature')
@click.option('--backend', type=click.Choice(['auto', 'openai', 'vllm', 'ollama', 'example']), 
              default='auto', help='Backend type')
@click.option('--api-key', help='API key for the backend')
@click.option('--base-url', help='Base URL for the backend')
def run(model: str, prompt_file: str, output: str, max_tokens: Optional[int], 
        temperature: Optional[float], backend: str, api_key: Optional[str], 
        base_url: Optional[str]):
    """Single-call inference: measure end-to-end latency and output."""
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if not prompts:
        click.echo("No prompts found in file", err=True)
        return
    
    # Load configuration
    config = load_config()
    
    # Get backend
    backend_kwargs = {}
    if api_key:
        backend_kwargs['api_key'] = api_key
    elif config.get('OPENAI_API_KEY') and config['OPENAI_API_KEY'] != 'your-api-key-here':
        backend_kwargs['api_key'] = config['OPENAI_API_KEY']
    
    if base_url:
        backend_kwargs['base_url'] = base_url
    elif config.get('OPENAI_BASE_URL'):
        backend_kwargs['base_url'] = config['OPENAI_BASE_URL']
    
    llm_backend = get_backend(model, backend, **backend_kwargs)
    
    # Run inference
    async def run_inference():
        results = []
        for i, prompt in enumerate(prompts):
            click.echo(f"Processing prompt {i+1}/{len(prompts)}...")
            result = await llm_backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert to dict for serialization
            result_dict = {
                'prompt': prompt,
                'text': result.text,
                'total_tokens': result.total_tokens,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens,
                'total_latency_ms': result.total_latency_ms,
                'first_token_latency_ms': result.first_token_latency_ms,
                'cost': result.cost,
                'model_name': result.model_name
            }
            results.append(result_dict)
        
        return results
    
    # Execute async function
    results = asyncio.run(run_inference())
    
    # Save results
    output_path = Path(output)
    if output_path.suffix.lower() == '.json':
        save_results_json(results, output)
    elif output_path.suffix.lower() == '.csv':
        save_results_csv(results, output)
    else:
        click.echo("Output file must have .json or .csv extension", err=True)
        return
    
    click.echo(f"Results saved to {output}")
    click.echo(f"Processed {len(results)} prompts")


@cli.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--prompt-file', '-f', required=True, help='File containing prompts (one per line)')
@click.option('--output', '-o', required=True, help='Output file (.json or .csv)')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, help='Sampling temperature')
@click.option('--backend', type=click.Choice(['auto', 'openai', 'vllm', 'ollama', 'example']), 
              default='auto', help='Backend type')
@click.option('--api-key', help='API key for the backend')
@click.option('--base-url', help='Base URL for the backend')
def profile(model: str, prompt_file: str, output: str, max_tokens: Optional[int], 
            temperature: Optional[float], backend: str, api_key: Optional[str], 
            base_url: Optional[str]):
    """Token-level profiling: capture per-token latencies."""
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if not prompts:
        click.echo("No prompts found in file", err=True)
        return
    
    # Load configuration
    config = load_config()
    
    # Get backend
    backend_kwargs = {}
    if api_key:
        backend_kwargs['api_key'] = api_key
    elif config.get('OPENAI_API_KEY') and config['OPENAI_API_KEY'] != 'your-api-key-here':
        backend_kwargs['api_key'] = config['OPENAI_API_KEY']
    
    if base_url:
        backend_kwargs['base_url'] = base_url
    elif config.get('OPENAI_BASE_URL'):
        backend_kwargs['base_url'] = config['OPENAI_BASE_URL']
    
    llm_backend = get_backend(model, backend, **backend_kwargs)
    
    # Run profiling
    async def run_profiling():
        results = []
        for i, prompt in enumerate(prompts):
            click.echo(f"Profiling prompt {i+1}/{len(prompts)}...")
            
            # Collect streaming tokens
            token_metrics = []
            async for token_metric in llm_backend.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                token_metrics.append(token_metric)
            
            # Convert to dict for serialization
            result_dict = {
                'prompt': prompt,
                'total_tokens': len(token_metrics),
                'token_metrics': [
                    {
                        'token': tm.token,
                        'timestamp': tm.timestamp,
                        'latency_ms': tm.latency_ms
                    }
                    for tm in token_metrics
                ],
                'model_name': model
            }
            results.append(result_dict)
        
        return results
    
    # Execute async function
    results = asyncio.run(run_profiling())
    
    # Save results
    output_path = Path(output)
    if output_path.suffix.lower() == '.json':
        save_results_json(results, output)
    elif output_path.suffix.lower() == '.csv':
        # Flatten token metrics for CSV
        flattened_results = []
        for result in results:
            for i, token_metric in enumerate(result['token_metrics']):
                flattened_results.append({
                    'prompt': result['prompt'],
                    'token_index': i,
                    'token': token_metric['token'],
                    'timestamp': token_metric['timestamp'],
                    'latency_ms': token_metric['latency_ms'],
                    'model_name': result['model_name']
                })
        save_results_csv(flattened_results, output)
    else:
        click.echo("Output file must have .json or .csv extension", err=True)
        return
    
    click.echo(f"Profiling results saved to {output}")
    click.echo(f"Profiled {len(results)} prompts")


@cli.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--prompt-file', '-f', required=True, help='File containing prompts (one per line)')
@click.option('--output', '-o', required=True, help='Output file (.json or .csv)')
@click.option('--batch-size', type=int, default=4, help='Batch size for processing')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, help='Sampling temperature')
@click.option('--backend', type=click.Choice(['auto', 'openai', 'vllm', 'ollama', 'example']), 
              default='auto', help='Backend type')
@click.option('--api-key', help='API key for the backend')
@click.option('--base-url', help='Base URL for the backend')
def batch(model: str, prompt_file: str, output: str, batch_size: int, 
          max_tokens: Optional[int], temperature: Optional[float], backend: str, 
          api_key: Optional[str], base_url: Optional[str]):
    """Batch inference: test throughput vs. latency trade-offs."""
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if not prompts:
        click.echo("No prompts found in file", err=True)
        return
    
    # Load configuration
    config = load_config()
    
    # Get backend
    backend_kwargs = {}
    if api_key:
        backend_kwargs['api_key'] = api_key
    elif config.get('OPENAI_API_KEY') and config['OPENAI_API_KEY'] != 'your-api-key-here':
        backend_kwargs['api_key'] = config['OPENAI_API_KEY']
    
    if base_url:
        backend_kwargs['base_url'] = base_url
    elif config.get('OPENAI_BASE_URL'):
        backend_kwargs['base_url'] = config['OPENAI_BASE_URL']
    
    llm_backend = get_backend(model, backend, **backend_kwargs)
    
    # Run batch inference
    async def run_batch():
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            click.echo(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}...")
            
            batch_result = await llm_backend.generate_batch(
                prompts=batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert to dict for serialization
            batch_dict = {
                'batch_index': i // batch_size,
                'batch_size': len(batch_prompts),
                'batch_latency_ms': batch_result.batch_latency_ms,
                'throughput_tokens_per_sec': batch_result.throughput_tokens_per_sec,
                'results': [
                    {
                        'prompt': prompt,
                        'text': result.text,
                        'total_tokens': result.total_tokens,
                        'input_tokens': result.input_tokens,
                        'output_tokens': result.output_tokens,
                        'total_latency_ms': result.total_latency_ms,
                        'cost': result.cost,
                        'model_name': result.model_name
                    }
                    for prompt, result in zip(batch_prompts, batch_result.results)
                ]
            }
            results.append(batch_dict)
        
        return results
    
    # Execute async function
    results = asyncio.run(run_batch())
    
    # Save results
    output_path = Path(output)
    if output_path.suffix.lower() == '.json':
        save_results_json(results, output)
    elif output_path.suffix.lower() == '.csv':
        # Flatten batch results for CSV
        flattened_results = []
        for batch_result in results:
            for result in batch_result['results']:
                flattened_results.append({
                    'batch_index': batch_result['batch_index'],
                    'batch_size': batch_result['batch_size'],
                    'batch_latency_ms': batch_result['batch_latency_ms'],
                    'throughput_tokens_per_sec': batch_result['throughput_tokens_per_sec'],
                    'prompt': result['prompt'],
                    'text': result['text'],
                    'total_tokens': result['total_tokens'],
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'total_latency_ms': result['total_latency_ms'],
                    'cost': result['cost'],
                    'model_name': result['model_name']
                })
        save_results_csv(flattened_results, output)
    else:
        click.echo("Output file must have .json or .csv extension", err=True)
        return
    
    click.echo(f"Batch results saved to {output}")
    click.echo(f"Processed {len(prompts)} prompts in {len(results)} batches")


@cli.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--prompt-file', '-f', required=True, help='File containing prompts (one per line)')
@click.option('--output', '-o', required=True, help='Output file (.json or .csv)')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, help='Sampling temperature')
@click.option('--backend', type=click.Choice(['auto', 'openai', 'vllm', 'ollama', 'example']), 
              default='auto', help='Backend type')
@click.option('--api-key', help='API key for the backend')
@click.option('--base-url', help='Base URL for the backend')
def stream(model: str, prompt_file: str, output: str, max_tokens: Optional[int], 
           temperature: Optional[float], backend: str, api_key: Optional[str], 
           base_url: Optional[str]):
    """Streaming mode: measure token-streaming latency buckets."""
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if not prompts:
        click.echo("No prompts found in file", err=True)
        return
    
    # Load configuration
    config = load_config()
    
    # Get backend
    backend_kwargs = {}
    if api_key:
        backend_kwargs['api_key'] = api_key
    elif config.get('OPENAI_API_KEY') and config['OPENAI_API_KEY'] != 'your-api-key-here':
        backend_kwargs['api_key'] = config['OPENAI_API_KEY']
    
    if base_url:
        backend_kwargs['base_url'] = base_url
    elif config.get('OPENAI_BASE_URL'):
        backend_kwargs['base_url'] = config['OPENAI_BASE_URL']
    
    llm_backend = get_backend(model, backend, **backend_kwargs)
    
    # Run streaming analysis
    async def run_streaming():
        results = []
        for i, prompt in enumerate(prompts):
            click.echo(f"Streaming prompt {i+1}/{len(prompts)}...")
            
            # Collect streaming tokens with latency buckets
            token_metrics = []
            latency_buckets = {
                '0-100ms': 0,
                '100-500ms': 0,
                '500ms-1s': 0,
                '1s-5s': 0,
                '5s+': 0
            }
            
            async for token_metric in llm_backend.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                token_metrics.append(token_metric)
                
                # Categorize latency
                latency = token_metric.latency_ms
                if latency < 100:
                    latency_buckets['0-100ms'] += 1
                elif latency < 500:
                    latency_buckets['100-500ms'] += 1
                elif latency < 1000:
                    latency_buckets['500ms-1s'] += 1
                elif latency < 5000:
                    latency_buckets['1s-5s'] += 1
                else:
                    latency_buckets['5s+'] += 1
            
            # Calculate statistics
            if token_metrics:
                latencies = [tm.latency_ms for tm in token_metrics]
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
            else:
                avg_latency = min_latency = max_latency = 0
            
            # Convert to dict for serialization
            result_dict = {
                'prompt': prompt,
                'total_tokens': len(token_metrics),
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'latency_buckets': latency_buckets,
                'token_metrics': [
                    {
                        'token': tm.token,
                        'timestamp': tm.timestamp,
                        'latency_ms': tm.latency_ms
                    }
                    for tm in token_metrics
                ],
                'model_name': model
            }
            results.append(result_dict)
        
        return results
    
    # Execute async function
    results = asyncio.run(run_streaming())
    
    # Save results
    output_path = Path(output)
    if output_path.suffix.lower() == '.json':
        save_results_json(results, output)
    elif output_path.suffix.lower() == '.csv':
        # Flatten streaming results for CSV
        flattened_results = []
        for result in results:
            for i, token_metric in enumerate(result['token_metrics']):
                flattened_results.append({
                    'prompt': result['prompt'],
                    'token_index': i,
                    'token': token_metric['token'],
                    'timestamp': token_metric['timestamp'],
                    'latency_ms': token_metric['latency_ms'],
                    'avg_latency_ms': result['avg_latency_ms'],
                    'min_latency_ms': result['min_latency_ms'],
                    'max_latency_ms': result['max_latency_ms'],
                    'latency_0_100ms': result['latency_buckets']['0-100ms'],
                    'latency_100_500ms': result['latency_buckets']['100-500ms'],
                    'latency_500ms_1s': result['latency_buckets']['500ms-1s'],
                    'latency_1s_5s': result['latency_buckets']['1s-5s'],
                    'latency_5s_plus': result['latency_buckets']['5s+'],
                    'model_name': result['model_name']
                })
        save_results_csv(flattened_results, output)
    else:
        click.echo("Output file must have .json or .csv extension", err=True)
        return
    
    click.echo(f"Streaming results saved to {output}")
    click.echo(f"Analyzed {len(results)} prompts")


if __name__ == '__main__':
    cli()
