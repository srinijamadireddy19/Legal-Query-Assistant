"""
Groq Provider
─────────────
LLM provider using Groq's ultra-fast inference API.

Supported models:
- llama-3.1-70b-versatile (recommended)
- llama-3.1-8b-instant (fast, less capable)
- llama-3.2-90b-vision-preview
- mixtral-8x7b-32768

Installation:
    pip install groq
"""

import logging
import time
from typing import List, Dict, Any

from ..core.models import LLMConfig, LLMResponse, ConversationMessage

log = logging.getLogger(__name__)


class GroqProvider:
    """
    LLM provider using Groq API.
    Handles chat completions with conversation history.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Groq client.
        
        Args:
            config: LLMConfig with API key and parameters
        """
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            
            if not self.config.api_key:
                raise ValueError("Groq API key is required")
            
            self.client = Groq(api_key=self.config.api_key)
            
            log.info(f"✓ Groq client initialized (model: {self.config.model_name})")
            
        except ImportError:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )
        except Exception as e:
            log.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Generate completion from Groq.
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            
        Returns:
            Response dict with content and metadata
        """
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
            )
            
            elapsed = time.time() - start_time
            
            # Extract response
            message = response.choices[0].message
            content = message.content
            
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_time_sec": elapsed,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }
            
        except Exception as e:
            log.error(f"Groq API call failed: {e}")
            raise
    
    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        history: List[ConversationMessage] = None,
    ) -> Dict[str, Any]:
        """
        Simple chat interface.
        
        Args:
            user_message: User's question
            system_prompt: Optional system prompt
            history: Optional conversation history
            
        Returns:
            Response dict
        """
        # Build messages
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history
        if history:
            messages.extend([msg.to_dict() for msg in history])
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return self.generate(messages)
    
    def is_available(self) -> bool:
        """Check if Groq API is available."""
        try:
            # Simple test call
            self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            log.warning(f"Groq API not available: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"GroqProvider(model={self.config.model_name})"


# ══════════════════════════════════════════════════════════════════════════
# Groq Model Presets
# ══════════════════════════════════════════════════════════════════════════

class GroqModels:
    """Preset configurations for popular Groq models."""
    
    @staticmethod
    def llama_70b_versatile(api_key: str) -> LLMConfig:
        """
        Llama 3.1 70B - Best balance of speed and quality.
        Recommended for production.
        """
        return LLMConfig(
            provider="groq",
            api_key=api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
        )
    
    @staticmethod
    def llama_8b_instant(api_key: str) -> LLMConfig:
        """
        Llama 3.1 8B - Ultra fast, good for simple queries.
        Use for high-throughput scenarios.
        """
        return LLMConfig(
            provider="groq",
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
        )
    
    @staticmethod
    def mixtral_8x7b(api_key: str) -> LLMConfig:
        """
        Mixtral 8x7B - Good alternative to Llama.
        32K context window.
        """
        return LLMConfig(
            provider="groq",
            api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=1024,
        )