"""
Enhanced AI agents for roop-unleashed with LLM and RAG integration.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from .base_agent import BaseAgent
from roop.llm_integration import LLMManager
from roop.rag_system import RAGSystem
from roop.error_handling import safe_execute, ErrorHandler

logger = logging.getLogger(__name__)


class EnhancedBaseAgent(BaseAgent):
    """Enhanced base agent with LLM and RAG capabilities."""
    
    def __init__(self, settings=None, llm_manager: Optional[LLMManager] = None, 
                 rag_system: Optional[RAGSystem] = None):
        self.settings = settings
        self.llm_manager = llm_manager
        self.rag_system = rag_system
        self.error_handler = ErrorHandler(settings) if settings else None
    
    def _query_llm(self, prompt: str, **kwargs) -> Optional[str]:
        """Query LLM with error handling."""
        if not self.llm_manager or not self.llm_manager.is_available():
            return None
        
        return safe_execute(
            self.llm_manager.generate,
            prompt,
            error_handler=self.error_handler,
            **kwargs
        )
    
    def _query_rag(self, question: str) -> Optional[str]:
        """Query RAG system with error handling."""
        if not self.rag_system:
            return None
        
        return safe_execute(
            self.rag_system.query,
            question,
            error_handler=self.error_handler
        )


class RAGAgent(EnhancedBaseAgent):
    """Agent specialized in knowledge retrieval and Q&A."""
    
    name: str = "rag"
    
    def assist(self, query: str) -> str:
        """Provide assistance using RAG system."""
        # First try RAG system
        rag_response = self._query_rag(query)
        if rag_response:
            return rag_response
        
        # Fallback to LLM if available
        if self.llm_manager and self.llm_manager.is_available():
            fallback_prompt = f"""
            You are a helpful assistant for the Roop face-swapping application. 
            Answer the following question based on your knowledge of face-swapping, 
            computer vision, and AI:
            
            Question: {query}
            
            Please provide a helpful, accurate response.
            """
            
            llm_response = self._query_llm(fallback_prompt, max_tokens=300)
            if llm_response:
                return f"Based on general knowledge: {llm_response}"
        
        return "I don't have specific information about that topic. Please check the documentation or try rephrasing your question."


class VideoProcessingAgent(EnhancedBaseAgent):
    """Agent specialized in video processing and editing assistance."""
    
    name: str = "video"
    
    def assist(self, query: str) -> str:
        """Provide video processing assistance."""
        # Check if it's a video-related query
        video_keywords = ['video', 'ffmpeg', 'codec', 'frame', 'fps', 'resolution', 'encoding']
        if not any(keyword in query.lower() for keyword in video_keywords):
            return "This query doesn't appear to be video-related. Try the 'rag' agent for general questions."
        
        # Try RAG first for specific video knowledge
        rag_response = self._query_rag(f"video processing {query}")
        if rag_response and len(rag_response) > 50:  # Substantial response
            return rag_response
        
        # Generate LLM response for video processing
        if self.llm_manager and self.llm_manager.is_available():
            prompt = f"""
            You are an expert in video processing for face-swapping applications. 
            The user has a question about video processing in the context of Roop face-swapping.
            
            Context:
            - Application: Roop face-swapping tool
            - Common formats: MP4, AVI, MOV
            - Common codecs: H.264, H.265, VP9
            - Typical issues: quality loss, artifacts, compatibility
            
            Question: {query}
            
            Provide specific, actionable advice for video processing in face-swapping:
            """
            
            response = self._query_llm(prompt, max_tokens=400)
            if response:
                return response
        
        return "For video processing questions, please ensure FFmpeg is installed and check the video format compatibility."


class OptimizationAgent(EnhancedBaseAgent):
    """Agent specialized in performance optimization advice."""
    
    name: str = "optimization"
    
    def assist(self, query: str) -> str:
        """Provide optimization assistance."""
        optimization_keywords = ['slow', 'fast', 'performance', 'gpu', 'cuda', 'memory', 'optimize', 'speed']
        if not any(keyword in query.lower() for keyword in optimization_keywords):
            return "This query doesn't appear to be optimization-related. Try the 'rag' agent for general questions."
        
        # Get system information if available
        system_info = self._get_system_info()
        
        # Try RAG first
        rag_response = self._query_rag(f"optimization performance {query}")
        if rag_response and len(rag_response) > 50:
            return rag_response
        
        # Generate optimization advice
        if self.llm_manager and self.llm_manager.is_available():
            prompt = f"""
            You are a performance optimization expert for AI face-swapping applications.
            
            System Context:
            {system_info}
            
            User Question: {query}
            
            Provide specific optimization recommendations for:
            1. GPU utilization (CUDA/ROCm)
            2. Memory management
            3. Processing speed improvements
            4. Quality vs speed tradeoffs
            
            Be specific and actionable:
            """
            
            response = self._query_llm(prompt, max_tokens=400)
            if response:
                return response
        
        return self._get_basic_optimization_advice()
    
    def _get_system_info(self) -> str:
        """Get basic system information for optimization advice."""
        info_parts = []
        
        if self.settings:
            provider = self.settings.provider
            max_threads = self.settings.max_threads
            memory_limit = self.settings.memory_limit
            
            info_parts.append(f"- GPU Provider: {provider}")
            info_parts.append(f"- Max Threads: {max_threads}")
            info_parts.append(f"- Memory Limit: {memory_limit}")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                info_parts.append(f"- CUDA GPUs: {gpu_count}")
                info_parts.append(f"- GPU Model: {gpu_name}")
        except Exception:
            pass
        
        return "\n".join(info_parts) if info_parts else "System information not available"
    
    def _get_basic_optimization_advice(self) -> str:
        """Provide basic optimization advice."""
        return """
        Basic Optimization Tips:
        1. Enable GPU acceleration (CUDA/ROCm) in settings
        2. Increase max_threads for CPU processing (4-8 recommended)
        3. Use appropriate video quality settings (14-20 for good balance)
        4. Enable face enhancement only when needed
        5. Process shorter video segments for memory efficiency
        6. Use SSD storage for temporary files
        7. Close other applications to free up GPU memory
        """


class ImageGenerationAgent(EnhancedBaseAgent):
    """Agent for AI image generation assistance."""
    
    name: str = "image_generation"
    
    def assist(self, query: str) -> str:
        """Provide image generation assistance."""
        generation_keywords = ['generate', 'create', 'image', 'picture', 'stable diffusion', 'dalle']
        if not any(keyword in query.lower() for keyword in generation_keywords):
            return "This query doesn't appear to be image generation-related."
        
        # Check if image generation is enabled
        if not self.settings or not self.settings.get_ai_setting('image_generation.enabled', False):
            return """
            Image generation is not currently enabled. To enable it:
            1. Set 'ai_config.image_generation.enabled' to true in configuration
            2. Configure the image generation provider (stable_diffusion, dalle)
            3. Ensure required models are downloaded
            4. Restart the application
            """
        
        # Try RAG for specific image generation knowledge
        rag_response = self._query_rag(f"image generation {query}")
        if rag_response:
            return rag_response
        
        # Generate advice using LLM
        if self.llm_manager and self.llm_manager.is_available():
            prompt = f"""
            You are an expert in AI image generation for face-swapping applications.
            
            Context:
            - Available providers: Stable Diffusion, DALL-E
            - Common use cases: Creating reference faces, backgrounds, training data
            - Integration with face-swapping workflow
            
            User Question: {query}
            
            Provide specific advice for image generation in the context of face-swapping:
            """
            
            response = self._query_llm(prompt, max_tokens=300)
            if response:
                return response
        
        return "Image generation requires proper configuration. Please check the AI settings and ensure models are available."


class TroubleshootingAgent(EnhancedBaseAgent):
    """Agent specialized in troubleshooting and error resolution."""
    
    name: str = "troubleshooting"
    
    def assist(self, query: str) -> str:
        """Provide troubleshooting assistance."""
        # Try RAG first for specific troubleshooting knowledge
        rag_response = self._query_rag(f"troubleshooting error {query}")
        if rag_response and len(rag_response) > 50:
            return rag_response
        
        # Analyze query for common error patterns
        error_analysis = self._analyze_error_query(query)
        
        if self.llm_manager and self.llm_manager.is_available():
            prompt = f"""
            You are a troubleshooting expert for the Roop face-swapping application.
            
            Common Issues:
            - GPU memory errors
            - Model loading failures
            - Video processing errors
            - Installation problems
            - Configuration issues
            
            Error Analysis: {error_analysis}
            
            User Problem: {query}
            
            Provide step-by-step troubleshooting instructions:
            """
            
            response = self._query_llm(prompt, max_tokens=400)
            if response:
                return response
        
        return self._get_basic_troubleshooting_steps(query)
    
    def _analyze_error_query(self, query: str) -> str:
        """Analyze the query for common error patterns."""
        query_lower = query.lower()
        
        if 'gpu' in query_lower or 'cuda' in query_lower or 'memory' in query_lower:
            return "GPU/Memory related issue"
        elif 'install' in query_lower or 'dependency' in query_lower:
            return "Installation issue"
        elif 'model' in query_lower or 'download' in query_lower:
            return "Model loading issue"
        elif 'video' in query_lower or 'ffmpeg' in query_lower:
            return "Video processing issue"
        elif 'config' in query_lower or 'setting' in query_lower:
            return "Configuration issue"
        else:
            return "General issue"
    
    def _get_basic_troubleshooting_steps(self, query: str) -> str:
        """Provide basic troubleshooting steps."""
        query_lower = query.lower()
        
        if 'gpu' in query_lower or 'memory' in query_lower:
            return """
            GPU/Memory Troubleshooting:
            1. Check GPU memory usage with nvidia-smi or rocm-smi
            2. Reduce batch size or frame buffer size in settings
            3. Close other GPU-using applications
            4. Try forcing CPU mode as fallback
            5. Restart the application to clear GPU memory
            """
        elif 'install' in query_lower:
            return """
            Installation Troubleshooting:
            1. Verify Python version (3.9-3.12)
            2. Install Visual C++ redistributables (Windows)
            3. Check CUDA/ROCm installation
            4. Update pip: pip install --upgrade pip
            5. Try clean virtual environment
            """
        else:
            return """
            General Troubleshooting:
            1. Check the logs directory for error details
            2. Verify all dependencies are installed
            3. Try with default configuration
            4. Restart the application
            5. Check GitHub issues for similar problems
            """


# Enhanced agent registry
ENHANCED_AGENTS = {
    RAGAgent.name: RAGAgent,
    VideoProcessingAgent.name: VideoProcessingAgent,
    OptimizationAgent.name: OptimizationAgent,
    ImageGenerationAgent.name: ImageGenerationAgent,
    TroubleshootingAgent.name: TroubleshootingAgent,
}