"""Ollama LLM integration - the AI's brain."""

import json
import ollama
from utils.logger import setup_logger

logger = setup_logger("Brain")


class Brain:
    def __init__(self, model: str = "llava"):
        """
        Initialize brain with Ollama model.
        IMPORTANT: Use a vision model (llava, bakllava, etc.)
        for screenshot understanding.
        """
        self.model = model
        self.conversation_history = []

    def analyze_screen(self, frame_base64: str, game_context: str,
                       available_actions: list) -> dict:
        """
        Send screenshot to LLM and get action decision.
        Returns: {"action": str, "reasoning": str, "observation": str}
        """
        actions_str = ", ".join(available_actions)

        prompt = f"""You are an AI playing a video game. Analyze this screenshot and decide what to do next.

GAME CONTEXT:
{game_context}

AVAILABLE ACTIONS: [{actions_str}]

Based on what you see in the screenshot, respond with ONLY valid JSON in this exact format:
{{
    "observation": "Brief description of what you see on screen",
    "reasoning": "Why you're choosing this action",
    "action": "exact_action_name_from_available_list",
    "action_sequence": ["action1", "action2"],
    "confidence": 0.8
}}

RULES:
1. ONLY use actions from the available actions list
2. action_sequence can contain 1-5 actions to perform in order
3. confidence is 0.0 to 1.0
4. Be specific in observations - mention health, enemies, items, menus, etc.
5. If you see a menu or death screen, handle it appropriately
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [frame_base64]
                    }
                ]
            )

            response_text = response["message"]["content"]
            logger.debug(f"Raw LLM response: {response_text}")

            # Parse JSON from response
            decision = self._parse_response(response_text)
            return decision

        except Exception as e:
            logger.error(f"Brain error: {e}")
            return {
                "observation": "Error processing screenshot",
                "reasoning": "Fallback action",
                "action": "wait",
                "action_sequence": ["wait"],
                "confidence": 0.0
            }

    def think(self, prompt: str) -> str:
        """General purpose thinking without screenshots."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Think error: {e}")
            return ""

    def plan_strategy(self, game_name: str, objective: str,
                      current_state: str) -> list:
        """Ask LLM to create a high-level strategy."""
        prompt = f"""You are playing {game_name}.
Your objective: {objective}
Current state: {current_state}

Create a step-by-step strategy. Respond with JSON:
{{
    "strategy": ["step1", "step2", "step3"],
    "sub_objectives": ["obj1", "obj2"],
    "estimated_difficulty": "easy/medium/hard"
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._parse_response(response["message"]["content"])
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            return {"strategy": ["observe", "act"], "sub_objectives": []}

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the response
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback
        logger.warning(f"Could not parse LLM response: {text[:200]}")
        return {
            "observation": text[:200],
            "reasoning": "Could not parse response",
            "action": "wait",
            "action_sequence": ["wait"],
            "confidence": 0.0
        }