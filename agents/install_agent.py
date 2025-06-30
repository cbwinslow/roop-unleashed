from .base_agent import BaseAgent


class InstallAgent(BaseAgent):
    name = "installer"

    def assist(self, query: str) -> str:
        return (
            "To install dependencies, run 'pip install -r requirements.txt'. "
            "Ensure you have a compatible GPU driver installed." 
        )
