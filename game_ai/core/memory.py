"""Memory system for tracking game state over time."""

import time
from collections import deque


class GameMemory:
    def __init__(self, max_states: int = 10):
        self.states = deque(maxlen=max_states)
        self.action_results = deque(maxlen=50)
        self.objectives = []
        self.current_objective = None
        self.knowledge = {}  # Persistent knowledge about the game

    def add_state(self, description: str, frame_base64: str = None):
        """Record a game state."""
        self.states.append({
            "description": description,
            "timestamp": time.time(),
            "frame": frame_base64
        })

    def add_action_result(self, action: str, result: str, success: bool):
        """Record what happened after an action."""
        self.action_results.append({
            "action": action,
            "result": result,
            "success": success,
            "timestamp": time.time()
        })

    def set_objective(self, objective: str):
        """Set current game objective."""
        self.current_objective = objective
        self.objectives.append(objective)

    def add_knowledge(self, key: str, value: str):
        """Store learned information about the game."""
        self.knowledge[key] = value

    def get_context_summary(self) -> str:
        """Build context summary for LLM."""
        summary_parts = []

        # Current objective
        if self.current_objective:
            summary_parts.append(
                f"CURRENT OBJECTIVE: {self.current_objective}"
            )

        # Recent states
        if self.states:
            summary_parts.append("RECENT OBSERVATIONS:")
            for state in list(self.states)[-5:]:
                summary_parts.append(f"  - {state['description']}")

        # Recent action results
        if self.action_results:
            summary_parts.append("RECENT ACTIONS & RESULTS:")
            for ar in list(self.action_results)[-5:]:
                status = "✓" if ar["success"] else "✗"
                summary_parts.append(
                    f"  {status} {ar['action']} → {ar['result']}"
                )

        # Knowledge base
        if self.knowledge:
            summary_parts.append("LEARNED KNOWLEDGE:")
            for k, v in self.knowledge.items():
                summary_parts.append(f"  {k}: {v}")

        return "\n".join(summary_parts)

    def clear(self):
        """Reset memory."""
        self.states.clear()
        self.action_results.clear()
        self.knowledge.clear()
        self.current_objective = None