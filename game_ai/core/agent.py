"""Main AI Agent - ties everything together."""

import time
from core.vision import Vision
from core.brain import Brain
from core.actions import ActionExecutor
from core.memory import GameMemory
from games.base_game import BaseGame
from utils.logger import setup_logger
from config import GAME_LOOP_DELAY, OLLAMA_MODEL

logger = setup_logger("Agent")


class GameAgent:
    def __init__(self, game_profile: BaseGame):
        self.game = game_profile
        self.vision = Vision()
        self.brain = Brain(model=OLLAMA_MODEL)
        self.actions = ActionExecutor()
        self.memory = GameMemory()
        self.running = False
        self.loop_count = 0
        self.start_time = None

        # Configure for the specific game
        self._setup_game()

    def _setup_game(self):
        """Configure agent for the specific game."""
        # Set up actions
        self.actions.setup_keyboard_actions(self.game.get_action_map())

        # Set screen region
        region = self.game.get_screen_region()
        if region:
            self.vision.set_region(
                region["left"], region["top"],
                region["width"], region["height"]
            )

        # Set objective
        self.memory.set_objective(self.game.get_objective())

        logger.info(f"Configured for: {self.game.name}")
        logger.info(f"Available actions: {self.actions.get_available_action_names()}")
        logger.info(f"Objective: {self.game.get_objective()}")

    def start(self, countdown: int = 5):
        """Start the game-playing loop with countdown."""
        print(f"\n{'='*50}")
        print(f"  GAME AI - Playing: {self.game.name}")
        print(f"  Model: {OLLAMA_MODEL}")
        print(f"  Objective: {self.game.get_objective()}")
        print(f"{'='*50}")
        print(f"\nSwitch to your game window now!")
        print(f"Starting in {countdown} seconds...")

        for i in range(countdown, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        print("GO! (Press Ctrl+C to stop)\n")

        self.running = True
        self.start_time = time.time()
        self.loop_count = 0

        try:
            self._game_loop()
        except KeyboardInterrupt:
            self.stop()

    def _game_loop(self):
        """Main game-playing loop."""
        while self.running:
            self.loop_count += 1
            loop_start = time.time()

            logger.info(f"--- Loop {self.loop_count} ---")

            # 1. CAPTURE - See the game
            frame_b64 = self.vision.get_base64_frame()

            # 2. BUILD CONTEXT
            context = self._build_context()

            # 3. THINK - Ask LLM what to do
            available = self.actions.get_available_action_names()
            decision = self.brain.analyze_screen(
                frame_b64, context, available
            )

            logger.info(f"Observation: {decision.get('observation', 'N/A')}")
            logger.info(f"Reasoning: {decision.get('reasoning', 'N/A')}")
            logger.info(f"Actions: {decision.get('action_sequence', [decision.get('action', 'wait')])}")
            logger.info(f"Confidence: {decision.get('confidence', 'N/A')}")

            # 4. ACT - Execute the decision
            action_sequence = decision.get(
                "action_sequence",
                [decision.get("action", "wait")]
            )

            for action_name in action_sequence:
                success = self.actions.execute(action_name)
                if not success:
                    logger.warning(f"Failed to execute: {action_name}")

            # 5. REMEMBER
            self.memory.add_state(decision.get("observation", ""))
            self.memory.add_action_result(
                str(action_sequence),
                decision.get("observation", ""),
                decision.get("confidence", 0) > 0.5
            )

            # 6. CHECK if screen changed (did our action do something?)
            if not self.vision.has_changed(threshold=0.98):
                logger.info("Screen hasn't changed much - action may not have worked")
                self.memory.add_knowledge(
                    f"loop_{self.loop_count}",
                    f"Action {action_sequence} may not have had effect"
                )

            # Timing
            loop_time = time.time() - loop_start
            logger.info(f"Loop time: {loop_time:.2f}s")

            # Wait before next loop
            wait_time = max(0, GAME_LOOP_DELAY - loop_time)
            time.sleep(wait_time)

    def _build_context(self) -> str:
        """Build context string for the LLM."""
        parts = [
            self.game.get_context(),
            "",
            self.memory.get_context_summary(),
            "",
            f"Game loop iteration: {self.loop_count}",
            f"Time playing: {time.time() - self.start_time:.0f}s"
        ]
        return "\n".join(parts)

    def stop(self):
        """Stop the agent."""
        self.running = False
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n{'='*50}")
        print(f"  AI Stopped")
        print(f"  Loops completed: {self.loop_count}")
        print(f"  Time played: {elapsed:.1f}s")
        print(f"  Actions taken: {len(self.memory.action_results)}")
        print(f"{'='*50}")