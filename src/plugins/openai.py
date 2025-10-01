"""OpenAI plugin implementation."""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import openai
from openai import AsyncOpenAI

from .base import BaseLLMBackend, InferenceResult, TokenMetrics, BatchResult


class OpenAIBackend(BaseLLMBackend):
    """OpenAI API backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the OpenAI backend.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        
        # Initialize OpenAI client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt using OpenAI API."""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            # Extract response data
            choice = response.choices[0]
            text = choice.message.content or ""
            
            # Calculate token counts
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost (rough estimation)
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return InferenceResult(
                text=text,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_ms=total_latency_ms,
                first_token_latency_ms=total_latency_ms * 0.1,  # Rough estimation
                model_name=self.model_name,
                cost=cost
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
        """Generate text with streaming token metrics using OpenAI API."""
        start_time = time.time()
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    current_time = time.time()
                    latency_ms = (current_time - start_time) * 1000
                    
                    yield TokenMetrics(
                        token=chunk.choices[0].delta.content,
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
        """Generate text for multiple prompts in batch using OpenAI API."""
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
            "backend": "openai",
            "base_url": self.base_url or "https://api.openai.com/v1",
            "supports_streaming": True,
            "supports_batching": True,
            "max_tokens": 4096,
            "context_length": 8192
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost based on token usage."""
        # Rough cost estimation (prices may vary)
        cost_per_1k_input = 0.0015  # $0.0015 per 1K input tokens
        cost_per_1k_output = 0.002  # $0.002 per 1K output tokens
        
        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        
        return input_cost + output_cost
