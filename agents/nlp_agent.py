import re
from typing import Tuple

from .base_agent import BaseAgent


class NLUModule:
    """Simple natural language understanding via regex rules."""

    patterns = {
        r"install|setup|dependency|gpu": "installer",
        r"model|weights|checkpoint": "model",
        r"process|operation|run|help": "operation",
    }

    @classmethod
    def parse(cls, query: str) -> Tuple[str, str]:
        """Return best matching agent name and cleaned query."""
        for pattern, agent in cls.patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                cleaned = re.sub(pattern, "", query, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                return agent, cleaned
        return "operation", query


class NLPAgent(BaseAgent):
    name = "nlu"

    def assist(self, query: str) -> str:
        agent, cleaned = NLUModule.parse(query)
        return f"NLU routed query to '{agent}' with payload: '{cleaned}'"
