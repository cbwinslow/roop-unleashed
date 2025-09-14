from .base_agent import BaseAgent


class InstallAgent(BaseAgent):
    name = "installer"

    def assist(self, query: str) -> str:
        return f"Install assistance for query: {query}"
