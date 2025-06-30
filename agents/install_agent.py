from .base_agent import BaseAgent


class InstallAgent(BaseAgent):
    name = "installer"

    def assist(self, query: str) -> str:
        lower = query.lower()
        if "gpu" in lower:
            return (
                "Install the latest GPU drivers and PyTorch with CUDA support. "
                "See https://pytorch.org for commands."
            )
        return (
            "To install dependencies, run 'pip install -r requirements.txt'. "
            "Ensure you have a compatible GPU driver installed."
        )
