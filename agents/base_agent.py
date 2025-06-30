class BaseAgent:
    """Basic interface for specialized helper agents."""

    name: str = "base"

    def assist(self, query: str) -> str:
        """Return a response for the given query."""
        raise NotImplementedError
