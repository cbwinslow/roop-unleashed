from __future__ import annotations
from typing import Dict, List, Optional
import logging

from .base_agent import BaseAgent
from .install_agent import InstallAgent
from .model_agent import ModelAgent
from .operation_agent import OperationAgent
from .nlp_agent import NLPAgent

try:
    from .enhanced_agents import ENHANCED_AGENTS
    from roop.llm_integration import LLMManager
    from roop.rag_system import RAGSystem
    from roop.nvidia_optimizer import NVIDIAOptimizer
    from roop.error_handling import ErrorHandler
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logging.warning(f"Enhanced AI features not available: {e}")

logger = logging.getLogger(__name__)


class MultiAgentManager:
    """Enhanced multi-agent manager with AI capabilities."""

    def __init__(self, settings=None) -> None:
        self.settings = settings
        self.error_handler = ErrorHandler(settings) if settings else None
        
        # Initialize core systems
        self.llm_manager = None
        self.rag_system = None
        self.nvidia_optimizer = None
        
        if ENHANCED_FEATURES_AVAILABLE and settings:
            self._initialize_enhanced_features()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = self._initialize_agents()
        
        logger.info(f"Initialized {len(self.agents)} agents")

    def _initialize_enhanced_features(self):
        """Initialize enhanced AI features."""
        try:
            # Initialize LLM manager
            if self.settings.get_ai_setting('local_llm.enabled', False):
                self.llm_manager = LLMManager(self.settings)
                if self.llm_manager.is_available():
                    logger.info("LLM manager initialized successfully")
                else:
                    logger.warning("LLM manager initialized but not available")
            
            # Initialize RAG system
            if self.settings.get_ai_setting('rag.enabled', False):
                self.rag_system = RAGSystem(self.settings, self.llm_manager)
                logger.info("RAG system initialized")
            
            # Initialize NVIDIA optimizer
            self.nvidia_optimizer = NVIDIAOptimizer(self.settings)
            logger.info("NVIDIA optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced features: {e}")
            if self.error_handler:
                self.error_handler.handle_error(e, "enhanced_features_init")

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all available agents."""
        agents = {}
        
        # Initialize core agents
        core_agents = {
            InstallAgent.name: InstallAgent(),
            ModelAgent.name: ModelAgent(),
            OperationAgent.name: OperationAgent(),
            NLPAgent.name: NLPAgent(),
        }
        
        for name, agent in core_agents.items():
            agents[name] = agent
            logger.debug(f"Initialized core agent: {name}")
        
        # Initialize enhanced agents if available
        if ENHANCED_FEATURES_AVAILABLE and self.settings:
            for name, agent_class in ENHANCED_AGENTS.items():
                try:
                    agent = agent_class(
                        settings=self.settings,
                        llm_manager=self.llm_manager,
                        rag_system=self.rag_system
                    )
                    agents[name] = agent
                    logger.debug(f"Initialized enhanced agent: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize enhanced agent {name}: {e}")
        
        return agents

    def available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())

    def assist(self, agent_name: str, query: str) -> str:
        """Get assistance from a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Unknown agent '{agent_name}'. Available agents: {', '.join(self.available_agents())}"
        
        try:
            response = agent.assist(query)
            logger.debug(f"Agent '{agent_name}' responded to query")
            return response
        except Exception as e:
            error_msg = f"Error getting assistance from agent '{agent_name}': {str(e)}"
            logger.error(error_msg)
            if self.error_handler:
                self.error_handler.handle_error(e, f"agent_assist_{agent_name}")
            return error_msg

    def smart_assist(self, query: str) -> str:
        """Automatically route query to the most appropriate agent."""
        # First try NLP agent for routing if available
        if 'nlp' in self.agents:
            try:
                nlp_agent = self.agents['nlp']
                if hasattr(nlp_agent, 'route_query'):
                    suggested_agent = nlp_agent.route_query(query)
                    if suggested_agent and suggested_agent in self.agents:
                        return self.assist(suggested_agent, query)
            except Exception as e:
                logger.debug(f"NLP routing failed: {e}")
        
        # Fallback to keyword-based routing
        agent_name = self._route_by_keywords(query)
        return self.assist(agent_name, query)

    def _route_by_keywords(self, query: str) -> str:
        """Route query based on keywords."""
        query_lower = query.lower()
        
        # Check for specific agent keywords
        if any(word in query_lower for word in ['video', 'ffmpeg', 'codec', 'frame']):
            return 'video' if 'video' in self.agents else 'operation'
        
        if any(word in query_lower for word in ['slow', 'fast', 'performance', 'gpu', 'optimize']):
            return 'optimization' if 'optimization' in self.agents else 'operation'
        
        if any(word in query_lower for word in ['error', 'problem', 'issue', 'troubleshoot']):
            return 'troubleshooting' if 'troubleshooting' in self.agents else 'operation'
        
        if any(word in query_lower for word in ['generate', 'create', 'image']):
            return 'image_generation' if 'image_generation' in self.agents else 'operation'
        
        if any(word in query_lower for word in ['install', 'dependency', 'requirement']):
            return 'installer'
        
        if any(word in query_lower for word in ['model', 'download', 'onnx']):
            return 'model'
        
        # Default to RAG agent for general questions, then operation
        if 'rag' in self.agents:
            return 'rag'
        return 'operation'

    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status."""
        status = {
            'agent_count': len(self.agents),
            'available_agents': self.available_agents(),
            'enhanced_features': ENHANCED_FEATURES_AVAILABLE
        }
        
        if self.llm_manager:
            status['llm_status'] = self.llm_manager.health_check()
        
        if self.rag_system:
            status['rag_status'] = self.rag_system.get_stats()
        
        if self.nvidia_optimizer:
            status['nvidia_status'] = self.nvidia_optimizer.get_status()
        
        return status

    def get_help_text(self) -> str:
        """Get help text for available agents and features."""
        help_lines = [
            "Roop Unleashed Multi-Agent Assistant",
            "=" * 40,
            "",
            "Available Agents:"
        ]
        
        agent_descriptions = {
            'installer': 'Installation and dependency management',
            'model': 'Model download and management',
            'operation': 'General face-swapping operations',
            'nlp': 'Natural language processing and routing',
            'rag': 'Knowledge base search and Q&A',
            'video': 'Video processing and editing assistance',
            'optimization': 'Performance optimization advice',
            'image_generation': 'AI image generation assistance',
            'troubleshooting': 'Error diagnosis and resolution'
        }
        
        for agent_name in self.available_agents():
            description = agent_descriptions.get(agent_name, 'Specialized assistance')
            help_lines.append(f"  {agent_name}: {description}")
        
        help_lines.extend([
            "",
            "Usage:",
            "  assist('<agent_name>', '<your_question>')",
            "  smart_assist('<your_question>')  # Auto-routes to best agent",
            "",
            "Examples:",
            "  assist('optimization', 'How to make processing faster?')",
            "  smart_assist('My video processing is slow')",
            "  assist('rag', 'What are the best face swapping practices?')"
        ])
        
        if ENHANCED_FEATURES_AVAILABLE:
            help_lines.extend([
                "",
                "Enhanced Features:",
                f"  LLM Available: {self.llm_manager.is_available() if self.llm_manager else False}",
                f"  RAG Enabled: {bool(self.rag_system)}",
                f"  NVIDIA Optimization: {bool(self.nvidia_optimizer)}"
            ])
        
        return "\n".join(help_lines)

    def add_knowledge(self, content: str, source: str = "user") -> bool:
        """Add knowledge to the RAG system."""
        if not self.rag_system:
            return False
        
        return self.rag_system.add_knowledge(content, {'source': source})

    def shutdown(self):
        """Cleanup and shutdown all systems."""
        try:
            if self.rag_system:
                self.rag_system.vector_store.save()
                logger.info("Saved RAG system data")
            
            if self.nvidia_optimizer:
                # Could add cleanup for CUDA contexts here
                pass
            
            logger.info("Multi-agent manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

