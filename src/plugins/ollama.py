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
        ollama_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt using Ollama API."""
        start_time = time.time()
        process = psutil.Process() if psutil else None
        
        # Sample CPU/memory before the call
        cpu_percent_before = None
        mem_rss_mb_before = None
        if process:
            try:
                cpu_percent_before = process.cpu_percent(interval=None)
                mem_rss_mb_before = process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        
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
        if ollama_options:
            payload["options"] = payload.get("options", {})
            payload["options"].update(ollama_options)
        
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

                        # Sample CPU/memory after the call and CPU times
                        cpu_percent_after = None
                        mem_rss_mb_after = None
                        cpu_user_s = None
                        cpu_system_s = None
                        if process:
                            try:
                                cpu_percent_after = process.cpu_percent(interval=None)
                                mem_rss_mb_after = process.memory_info().rss / (1024 * 1024)
                                ct = process.cpu_times()
                                cpu_user_s = getattr(ct, 'user', 0.0)
                                cpu_system_s = getattr(ct, 'system', 0.0)
                            except Exception:
                                pass
                        
                        # Use the higher of before/after for CPU, last observed for memory
                        cpu_percent = cpu_percent_after if cpu_percent_after is not None else cpu_percent_before
                        mem_rss_mb = mem_rss_mb_after if mem_rss_mb_after is not None else mem_rss_mb_before
                        
                        # Estimate first token latency (prompt processing time)
                        first_token_latency_ms = None
                        if isinstance(prompt_eval_duration_ns, int) and prompt_eval_duration_ns > 0:
                            first_token_latency_ms = prompt_eval_duration_ns / 1e6  # ns -> ms

                        # Prompt-side tokens/sec
                        prompt_tokens_per_sec = None
                        if isinstance(prompt_eval_duration_ns, int) and prompt_eval_duration_ns > 0 and input_tokens > 0:
                            prompt_tokens_per_sec = (input_tokens / (prompt_eval_duration_ns / 1e9))

                        # Estimate KV cache memory and context utilization if num_ctx provided
                        kv_cache_mb_estimate = None
                        ctx_utilization = None
                        num_ctx = None
                        opts = payload.get('options', {})
                        if isinstance(opts, dict) and 'num_ctx' in opts:
                            try:
                                num_ctx = int(opts['num_ctx'])
                            except Exception:
                                num_ctx = None
                        if isinstance(num_ctx, int) and num_ctx > 0:
                            try:
                                bytes_per_token = 2048  # heuristic placeholder for comparison across runs
                                ctx_utilization = min(1.0, (total_tokens / max(1, num_ctx)))
                                kv_cache_mb_estimate = (bytes_per_token * num_ctx) / (1024 * 1024)
                            except Exception:
                                kv_cache_mb_estimate = None

                        return InferenceResult(
                            text=text,
                            total_tokens=total_tokens,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_latency_ms=total_latency_ms,
                            first_token_latency_ms=first_token_latency_ms,
                            model_name=self.model_name,
                            cost=0.0,
                            tokens_per_sec=tokens_per_sec,
                            cpu_percent=cpu_percent,
                            mem_rss_mb=mem_rss_mb,
                            prompt_tokens_per_sec=prompt_tokens_per_sec,
                            kv_cache_mb_estimate=kv_cache_mb_estimate,
                            ctx_utilization=ctx_utilization,
                            cpu_user_s=cpu_user_s,
                            cpu_system_s=cpu_system_s
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
