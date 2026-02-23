"""Base game profile - inherit this for specific games."""

from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self):
        self.name = "Unknown Game"
        self.action_map = {}
        self.screen_region = None  # None = full screen
        self.objective = "Play the game"
        self.game_specific_context = ""

    @abstractmethod
    def get_action_map(self) -> dict:
        """Return game-specific action mapping."""
        pass

    @abstractmethod
    def get_objective(self) -> str:
        """Return the game's primary objective."""
        pass

    def get_screen_region(self) -> dict:
        """Return screen capture region or None for fullscreen."""
        return self.screen_region

    def get_context(self) -> str:
        """Return game-specific context for the LLM."""
        return self.game_specific_context

    def get_system_prompt(self) -> str:
        """Return game-specific system prompt."""
        return f"You are playing {self.name}. {self.game_specific_context}"