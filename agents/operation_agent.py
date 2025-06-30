from .base_agent import BaseAgent


class OperationAgent(BaseAgent):
    name = "operation"

    def assist(self, query: str) -> str:
        return (
            "Use the GUI tabs to configure inputs and choose processors. "
            "Adjust parameters like blend_ratio for better results." 
        )
