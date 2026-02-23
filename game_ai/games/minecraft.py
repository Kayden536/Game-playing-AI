from games.base_game import BaseGame


class MinecraftProfile(BaseGame):
    def __init__(self):
        super().__init__()
        self.name = "Minecraft"
        self.game_specific_context = """
        This is Minecraft, a 3D survival/sandbox game.
        - Health bar is at the bottom (hearts)
        - Hunger bar is at the bottom (drumsticks)
        - Hotbar shows inventory slots at the bottom
        - Crosshair is in the center of screen
        - Day/night cycle affects gameplay (monsters at night)
        - You need to gather resources, craft tools, build shelter
        - Watch for Creepers (green), Zombies, Skeletons, Spiders
        """

    def get_action_map(self) -> dict:
        return {
            "move_forward": {"type": "key_hold", "key": "w"},
            "move_backward": {"type": "key_hold", "key": "s"},
            "move_left": {"type": "key_hold", "key": "a"},
            "move_right": {"type": "key_hold", "key": "d"},
            "jump": {"type": "key_press", "key": "space"},
            "attack": {"type": "mouse_click", "button": "left"},
            "use_item": {"type": "mouse_click", "button": "right"},
            "inventory": {"type": "key_press", "key": "e"},
            "sprint": {"type": "key_hold", "key": "ctrl"},
            "sneak": {"type": "key_hold", "key": "shift"},
            "drop": {"type": "key_press", "key": "q"},
            "slot_1": {"type": "key_press", "key": "1"},
            "slot_2": {"type": "key_press", "key": "2"},
            "slot_3": {"type": "key_press", "key": "3"},
            "slot_4": {"type": "key_press", "key": "4"},
            "slot_5": {"type": "key_press", "key": "5"},
            "look_left": {"type": "mouse_move", "x": -100, "y": 0},
            "look_right": {"type": "mouse_move", "x": 100, "y": 0},
            "look_up": {"type": "mouse_move", "x": 0, "y": -50},
            "look_down": {"type": "mouse_move", "x": 0, "y": 50},
            "wait": {"type": "wait", "duration": 0.5},
            "escape": {"type": "key_press", "key": "escape"},
        }

    def get_objective(self) -> str:
        return "Survive. Gather resources, craft tools, build a shelter before nightfall."