"""Input execution module - handles all game inputs."""

import time
import pydirectinput
from config import INPUT_DELAY

# Disable pydirectinput pause
pydirectinput.PAUSE = 0.0


class ActionExecutor:
    def __init__(self):
        self.available_actions = {}
        self.action_history = []

    def setup_keyboard_actions(self, action_map: dict = None):
        """
        Setup available keyboard actions.
        Default provides common game controls.
        """
        if action_map is None:
            action_map = {
                "move_forward": {"type": "key_hold", "key": "w"},
                "move_backward": {"type": "key_hold", "key": "s"},
                "move_left": {"type": "key_hold", "key": "a"},
                "move_right": {"type": "key_hold", "key": "d"},
                "jump": {"type": "key_press", "key": "space"},
                "attack": {"type": "mouse_click", "button": "left"},
                "use": {"type": "mouse_click", "button": "right"},
                "inventory": {"type": "key_press", "key": "e"},
                "sprint": {"type": "key_hold", "key": "shift"},
                "sneak": {"type": "key_hold", "key": "ctrl"},
                "drop": {"type": "key_press", "key": "q"},
                "chat": {"type": "key_press", "key": "t"},
                "escape": {"type": "key_press", "key": "escape"},
                "click": {"type": "mouse_click", "button": "left"},
                "wait": {"type": "wait", "duration": 0.5},
                # Hotbar slots
                "slot_1": {"type": "key_press", "key": "1"},
                "slot_2": {"type": "key_press", "key": "2"},
                "slot_3": {"type": "key_press", "key": "3"},
                "slot_4": {"type": "key_press", "key": "4"},
                "slot_5": {"type": "key_press", "key": "5"},
            }
        self.available_actions = action_map

    def execute(self, action_name: str, duration: float = 0.3) -> bool:
        """Execute a named action."""
        if action_name not in self.available_actions:
            print(f"Unknown action: {action_name}")
            return False

        action = self.available_actions[action_name]
        action_type = action["type"]

        try:
            if action_type == "key_press":
                pydirectinput.press(action["key"])

            elif action_type == "key_hold":
                pydirectinput.keyDown(action["key"])
                time.sleep(duration)
                pydirectinput.keyUp(action["key"])

            elif action_type == "mouse_click":
                if action["button"] == "left":
                    pydirectinput.click()
                elif action["button"] == "right":
                    pydirectinput.rightClick()

            elif action_type == "mouse_move":
                pydirectinput.moveRel(
                    action.get("x", 0),
                    action.get("y", 0)
                )

            elif action_type == "wait":
                time.sleep(action.get("duration", 0.5))

            self.action_history.append({
                "action": action_name,
                "timestamp": time.time()
            })

            time.sleep(INPUT_DELAY)
            return True

        except Exception as e:
            print(f"Action execution error: {e}")
            return False

    def execute_sequence(self, actions: list, delay: float = 0.2):
        """Execute multiple actions in sequence."""
        for action_name in actions:
            self.execute(action_name)
            time.sleep(delay)

    def mouse_look(self, x_offset: int, y_offset: int):
        """Move mouse for camera control (3D games)."""
        pydirectinput.moveRel(x_offset, y_offset, relative=True)

    def click_at(self, x: int, y: int):
        """Click at specific screen coordinates."""
        pydirectinput.click(x=x, y=y)

    def get_available_action_names(self) -> list:
        """Return list of available action names."""
        return list(self.available_actions.keys())