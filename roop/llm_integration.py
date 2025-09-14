"""
Local LLM integration for roop-unleashed.
Supports Ollama and OpenAI-compatible local APIs.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Iterator
from abc import ABC, abstractmethod

from .error_handling import retry_on_error, RoopException

logger = logging.getLogger(__name__)


class LLMError(RoopException):
    """Raised when LLM operations fail."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout = config.get('timeout', 30)
        self.model = config.get('model', 'llama3.2:3b')
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = f"{self.base_url}/api"
    
    @retry_on_error(max_retries=2, delay=1.0, exceptions=(requests.RequestException,))
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    'num_predict': kwargs.get('max_tokens', self.config.get('max_tokens', 1000)),
                }
            }
            
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise LLMError(f"Failed to generate text with Ollama: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate text with streaming response."""
        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': True,
                'options': {
                    'temperature': kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    'num_predict': kwargs.get('max_tokens', self.config.get('max_tokens', 1000)),
                }
            }
            
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                        
        except requests.RequestException as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise LLMError(f"Failed to stream from Ollama: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Download a model if not available locally."""
        try:
            payload = {'name': model_name}
            response = requests.post(
                f"{self.api_base}/pull",
                json=payload,
                timeout=300  # Model downloads can take a while
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible API provider for local LLMs."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key', 'local')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    @retry_on_error(max_retries=2, delay=1.0, exceptions=(requests.RequestException,))
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI-compatible API."""
        try:
            payload = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': kwargs.get('temperature', self.config.get('temperature', 0.7)),
                'max_tokens': kwargs.get('max_tokens', self.config.get('max_tokens', 1000)),
                'stream': False
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.RequestException as e:
            logger.error(f"OpenAI-compatible API request failed: {e}")
            raise LLMError(f"Failed to generate text: {e}")
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise LLMError(f"Invalid API response: {e}")
    
    def is_available(self) -> bool:
        """Check if the API is available."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", 
                                  headers=self.headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", 
                                  headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


class LLMManager:
    """Manages LLM providers and provides a unified interface."""
    
    def __init__(self, settings):
        self.settings = settings
        self.provider = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate LLM provider based on configuration."""
        if not self.settings.get_ai_setting('local_llm.enabled', False):
            logger.info("Local LLM is disabled in configuration")
            return
        
        provider_type = self.settings.get_ai_setting('local_llm.provider', 'ollama')
        config = self.settings.get_ai_setting('local_llm', {})
        
        try:
            if provider_type == 'ollama':
                self.provider = OllamaProvider(config)
            elif provider_type == 'local_api':
                self.provider = OpenAICompatibleProvider(config)
            else:
                logger.error(f"Unknown LLM provider: {provider_type}")
                return
            
            if self.provider.is_available():
                logger.info(f"LLM provider {provider_type} initialized successfully")
            else:
                logger.warning(f"LLM provider {provider_type} is not available")
                self.provider = None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            self.provider = None
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return self.provider is not None and self.provider.is_available()
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using the configured LLM provider."""
        if not self.is_available():
            logger.warning("No LLM provider available")
            return None
        
        try:
            return self.provider.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
    
    def generate_stream(self, prompt: str, **kwargs) -> Optional[Iterator[str]]:
        """Generate text with streaming if supported."""
        if not self.is_available():
            return None
        
        if hasattr(self.provider, 'generate_stream'):
            try:
                return self.provider.generate_stream(prompt, **kwargs)
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")
                return None
        else:
            # Fallback to regular generation
            result = self.generate(prompt, **kwargs)
            if result:
                yield result
    
    def list_models(self) -> List[str]:
        """List available models."""
        if not self.is_available():
            return []
        return self.provider.list_models()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the LLM system."""
        if not self.provider:
            return {'status': 'disabled', 'available': False}
        
        try:
            available = self.provider.is_available()
            models = self.provider.list_models() if available else []
            
            # Try a simple generation to test functionality
            test_response = None
            if available:
                test_response = self.provider.generate("Hello", max_tokens=10)
            
            return {
                'status': 'healthy' if available and test_response else 'unhealthy',
                'available': available,
                'provider_type': type(self.provider).__name__,
                'model_count': len(models),
                'current_model': self.provider.model,
                'test_response_length': len(test_response) if test_response else 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'available': False,
                'error': str(e)
            }


# Utility functions for common LLM tasks
def generate_image_description(llm_manager: LLMManager, image_path: str) -> Optional[str]:
    """Generate a description for face-swapping context."""
    if not llm_manager.is_available():
        return None
    
    prompt = f"""
    You are helping with a face-swapping application. Given an image file at {image_path}, 
    provide a brief, professional description that would help with:
    1. Face detection quality assessment
    2. Lighting and angle considerations
    3. Recommended processing parameters
    
    Keep the response concise and technical.
    """
    
    return llm_manager.generate(prompt, max_tokens=200)


def generate_processing_advice(llm_manager: LLMManager, context: Dict[str, Any]) -> Optional[str]:
    """Generate advice for processing parameters based on context."""
    if not llm_manager.is_available():
        return None
    
    prompt = f"""
    You are an expert in face-swapping technology. Based on the following context:
    - Source image quality: {context.get('source_quality', 'unknown')}
    - Target image quality: {context.get('target_quality', 'unknown')}
    - Processing mode: {context.get('mode', 'unknown')}
    - GPU available: {context.get('gpu_available', False)}
    
    Provide specific recommendations for:
    1. Optimal processing parameters
    2. Expected processing time
    3. Quality enhancement suggestions
    
    Be concise and practical.
    """
    
    return llm_manager.generate(prompt, max_tokens=300)