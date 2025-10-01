"""vLLM plugin implementation."""

import asyncio
import time
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator

from .base import BaseLLMBackend, InferenceResult, TokenMetrics, BatchResult


class VLLMBackend(BaseLLMBackend):
    """vLLM API backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the vLLM backend.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:8000")
        self.api_key = kwargs.get("api_key")  # Optional for local vLLM
        
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt using vLLM API."""
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()
                        total_latency_ms = (end_time - start_time) * 1000
                        
                        # Extract response data
                        choice = data["choices"][0]
                        text = choice["message"]["content"]
                        
                        # Calculate token counts
                        usage = data.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        
                        return InferenceResult(
                            text=text,
                            total_tokens=total_tokens,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_latency_ms=total_latency_ms,
                            first_token_latency_ms=total_latency_ms * 0.1,  # Rough estimation
                            model_name=self.model_name,
                            cost=0.0  # Local inference is free
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
        """Generate text with streaming token metrics using vLLM API."""
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data_str = line[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                
                                try:
                                    import json
                                    data = json.loads(data_str)
                                    if "choices" in data and data["choices"]:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            current_time = time.time()
                                            latency_ms = (current_time - start_time) * 1000
                                            
                                            yield TokenMetrics(
                                                token=delta["content"],
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
        """Generate text for multiple prompts in batch using vLLM API."""
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
            "backend": "vllm",
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_batching": True,
            "max_tokens": 4096,
            "context_length": 8192
        }
