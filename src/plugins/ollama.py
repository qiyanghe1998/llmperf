"""Ollama plugin implementation."""

import asyncio
import time
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
try:
    import psutil  # optional, for CPU/memory sampling
except Exception:  # pragma: no cover
    psutil = None

from .base import BaseLLMBackend, InferenceResult, TokenMetrics, BatchResult


class OllamaBackend(BaseLLMBackend):
    """Ollama API backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the Ollama backend.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt using Ollama API."""
        start_time = time.time()
        process = psutil.Process() if psutil else None
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens
        if temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = temperature
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()
                        total_latency_ms = (end_time - start_time) * 1000
                        
                        # Extract response data
                        text = data.get("response", "")

                        # Use Ollama's returned metrics if present
                        prompt_eval_count = data.get("prompt_eval_count")
                        eval_count = data.get("eval_count")
                        prompt_eval_duration_ns = data.get("prompt_eval_duration")
                        eval_duration_ns = data.get("eval_duration")

                        input_tokens = int(prompt_eval_count) if isinstance(prompt_eval_count, int) else 0
                        output_tokens = int(eval_count) if isinstance(eval_count, int) else 0
                        total_tokens = input_tokens + output_tokens

                        # Tokens/sec from eval duration (output side)
                        tokens_per_sec = None
                        if isinstance(eval_duration_ns, int) and eval_duration_ns > 0 and output_tokens > 0:
                            tokens_per_sec = (output_tokens / (eval_duration_ns / 1e9))

                        cpu_percent = None
                        mem_rss_mb = None
                        if process:
                            try:
                                cpu_percent = process.cpu_percent(interval=None)
                                mem_rss_mb = process.memory_info().rss / (1024 * 1024)
                            except Exception:
                                pass
                        
                        return InferenceResult(
                            text=text,
                            total_tokens=total_tokens,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_latency_ms=total_latency_ms,
                            first_token_latency_ms=None,
                            model_name=self.model_name,
                            cost=0.0,  # Local inference is free
                            tokens_per_sec=tokens_per_sec,
                            cpu_percent=cpu_percent,
                            mem_rss_mb=mem_rss_mb
                        )
                    else:
                        error_text = await response.text()
                        end_time = time.time()
                        total_latency_ms = (end_time - start_time) * 1000
                        
                        return InferenceResult(
                            text=f"Error {response.status}: {error_text}",
                            total_tokens=0,
                            input_tokens=0,
                            output_tokens=0,
                            total_latency_ms=total_latency_ms,
                            model_name=self.model_name,
                            cost=0.0
                        )
                        
        except Exception as e:
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            return InferenceResult(
                text=f"Error: {str(e)}",
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                total_latency_ms=total_latency_ms,
                model_name=self.model_name,
                cost=0.0
            )
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[TokenMetrics, None]:
        """Generate text with streaming token metrics using Ollama API."""
        start_time = time.time()
        process = psutil.Process() if psutil else None
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens
        if temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = temperature
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                try:
                                    import json
                                    data = json.loads(line)
                                    if "response" in data:
                                        current_time = time.time()
                                        latency_ms = (current_time - start_time) * 1000
                                        yield TokenMetrics(
                                            token=data["response"],
                                            timestamp=current_time,
                                            latency_ms=latency_ms
                                        )
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        current_time = time.time()
                        latency_ms = (current_time - start_time) * 1000
                        
                        yield TokenMetrics(
                            token=f"Error {response.status}: {error_text}",
                            timestamp=current_time,
                            latency_ms=latency_ms
                        )
                        
        except Exception as e:
            current_time = time.time()
            latency_ms = (current_time - start_time) * 1000
            
            yield TokenMetrics(
                token=f"Error: {str(e)}",
                timestamp=current_time,
                latency_ms=latency_ms
            )
    
    async def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BatchResult:
        """Generate text for multiple prompts in batch using Ollama API."""
        start_time = time.time()
        
        # Process prompts in parallel
        tasks = [self.generate(prompt, max_tokens, temperature, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        batch_latency_ms = (end_time - start_time) * 1000
        
        # Calculate throughput
        total_tokens = sum(result.total_tokens for result in results)
        throughput_tokens_per_sec = (total_tokens / batch_latency_ms) * 1000 if batch_latency_ms > 0 else 0
        
        return BatchResult(
            results=results,
            batch_latency_ms=batch_latency_ms,
            throughput_tokens_per_sec=throughput_tokens_per_sec
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "backend": "ollama",
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_batching": True,
            "max_tokens": 4096,
            "context_length": 8192
        }
