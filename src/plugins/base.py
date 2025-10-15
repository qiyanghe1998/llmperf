"""Base plugin interface for LLM backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import time


@dataclass
class TokenMetrics:
    """Metrics for a single token."""
    token: str
    timestamp: float
    latency_ms: float


@dataclass
class InferenceResult:
    """Result of an inference call."""
    text: str
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    first_token_latency_ms: Optional[float] = None
    token_metrics: Optional[List[TokenMetrics]] = None
    cost: Optional[float] = None
    model_name: Optional[str] = None
    tokens_per_sec: Optional[float] = None
    cpu_percent: Optional[float] = None
    mem_rss_mb: Optional[float] = None
    prompt_tokens_per_sec: Optional[float] = None
    kv_cache_mb_estimate: Optional[float] = None
    ctx_utilization: Optional[float] = None
    cpu_user_s: Optional[float] = None
    cpu_system_s: Optional[float] = None


@dataclass
class BatchResult:
    """Result of a batch inference call."""
    results: List[InferenceResult]
    batch_latency_ms: float
    throughput_tokens_per_sec: float


class BaseLLMBackend(ABC):
    """Base class for LLM backend implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the backend.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional backend-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            InferenceResult with metrics
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[TokenMetrics, None]:
        """Generate text with streaming token metrics.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Yields:
            TokenMetrics for each token
        """
        pass
    
    @abstractmethod
    async def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BatchResult:
        """Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            BatchResult with metrics
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def _calculate_token_latency(self, start_time: float, end_time: float, token_count: int) -> float:
        """Calculate average latency per token in milliseconds."""
        if token_count == 0:
            return 0.0
        return (end_time - start_time) * 1000 / token_count
