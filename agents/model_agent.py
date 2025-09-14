from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    name = "model"

    def assist(self, query: str) -> str:
        return f"Model assistance for query: {query}"
