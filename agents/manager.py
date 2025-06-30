from __future__ import annotations
from typing import Dict, List

from .base_agent import BaseAgent
from .install_agent import InstallAgent
from .model_agent import ModelAgent
from .operation_agent import OperationAgent
from .nlp_agent import NLPAgent


class MultiAgentManager:
    """Simple registry that dispatches queries to specialized agents."""

    def __init__(self) -> None:
        self.agents: Dict[str, BaseAgent] = {
            InstallAgent.name: InstallAgent(),
            ModelAgent.name: ModelAgent(),
            OperationAgent.name: OperationAgent(),
            NLPAgent.name: NLPAgent(),
        }

    def available_agents(self) -> List[str]:
        return list(self.agents.keys())

    def assist(self, agent_name: str, query: str) -> str:
        agent = self.agents.get(agent_name)
        if not agent:
            return "Unknown agent"
        return agent.assist(query)

