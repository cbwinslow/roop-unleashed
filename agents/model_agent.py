from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    name = "model"

    MODELS = [
        "ninjawick/webui-faceswap-unlocked",
        "PiAPI/Faceswap-API",
        "supArs/face_swap",
        "tsi-org/face-swapper",
        "kiddobellamy/faceswapping_kiddo",
    ]

    def assist(self, query: str) -> str:
        lower = query.lower()
        if "list" in lower or "recommend" in lower:
            return "Recommended models:\n" + "\n".join(f"- {m}" for m in self.MODELS)
        return (
            "You can place model files in the 'models' directory. "
            "Use the settings tab to select your preferred model."
        )
