from __future__ import annotations

from agents import MultiAgentManager


class InteractiveHelp:
    """Simple CLI for interacting with helper agents."""

    def __init__(self) -> None:
        self.manager = MultiAgentManager()

    def run(self) -> None:
        print("Interactive help. Type 'quit' to exit.")

        while True:
            query = input('> ')
            if query.lower() in {"quit", "exit"}:
                break

            response = self.manager.assist(agent.strip(), text.strip())
            print(response)


if __name__ == "__main__":
    InteractiveHelp().run()
