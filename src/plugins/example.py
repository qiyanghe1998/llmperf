"""Example plugin template for implementing new LLM backends."""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator

from .base import BaseLLMBackend, InferenceResult, TokenMetrics, BatchResult


class ExampleBackend(BaseLLMBackend):
    """Example backend implementation for demonstration purposes."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the example backend.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url", "https://api.example.com")
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt.
        
        This is a mock implementation that simulates API calls.
        Replace with actual API calls to your LLM service.
        """
        start_time = time.time()
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Mock response
        mock_response = f"Mock response to: {prompt[:50]}..."
        end_time = time.time()
        
        # Calculate metrics
        total_latency_ms = (end_time - start_time) * 1000
        input_tokens = len(prompt.split())  # Rough estimation
        output_tokens = len(mock_response.split())  # Rough estimation
        
        return InferenceResult(
            text=mock_response,
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_latency_ms=total_latency_ms,
            first_token_latency_ms=total_latency_ms * 0.1,  # Mock first token latency
            model_name=self.model_name,
            cost=0.001  # Mock cost
        )
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[TokenMetrics, None]:
        """Generate text with streaming token metrics.
        
        This is a mock implementation that simulates streaming.
        Replace with actual streaming API calls.
        """
        start_time = time.time()
        
        # Mock streaming tokens
        mock_tokens = ["Hello", " world", "!", " This", " is", " a", " mock", " response."]
        
        for i, token in enumerate(mock_tokens):
            # Simulate token generation delay
            await asyncio.sleep(0.05)
            
            current_time = time.time()
            latency_ms = (current_time - start_time) * 1000
            
            yield TokenMetrics(
                token=token,
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
        """Generate text for multiple prompts in batch.
        
        This is a mock implementation that simulates batch processing.
        Replace with actual batch API calls.
        """
        start_time = time.time()
        
        # Process prompts in parallel (mock)
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
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "backend": "example",
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_batching": True,
            "max_tokens": 4096,
            "context_length": 8192
        }
