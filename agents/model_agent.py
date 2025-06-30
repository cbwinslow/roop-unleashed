from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    name = "model"

    def assist(self, query: str) -> str:
        return (
            "You can place model files in the 'models' directory. "
            "Use the settings tab to select your preferred model." 
        )
