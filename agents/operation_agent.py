from .base_agent import BaseAgent


class OperationAgent(BaseAgent):
    name = "operation"

    def assist(self, query: str) -> str:
        return f"Operation assistance for query: {query}"
