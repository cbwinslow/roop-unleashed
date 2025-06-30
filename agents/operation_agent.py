from .base_agent import BaseAgent


class OperationAgent(BaseAgent):
    name = "operation"

    def assist(self, query: str) -> str:
        if "help" in query.lower():
            return (
                "Open the Extras tab for video tools or the Settings tab to "
                "adjust parameters like blend_ratio."
            )
        return (
            "Use the GUI tabs to configure inputs and choose processors. "
            "Adjust parameters like blend_ratio for better results."
        )
