from games.base_game import BaseGame


class FlappyBirdProfile(BaseGame):
    def __init__(self):
        super().__init__()
        self.name = "Flappy Bird"
        self.game_specific_context = """
        This is Flappy Bird, a simple 2D game.
        - The bird is on the left side of the screen
        - Green pipes come from top and bottom with gaps
        - Press space/click to flap (go up)
        - The bird falls due to gravity
        - Avoid hitting pipes and the ground
        - Time your flaps to pass through gaps
        - If the bird is above the gap, DON'T flap
        - If the bird is below the gap, flap
        """

    def get_action_map(self) -> dict:
        return {
            "flap": {"type": "key_press", "key": "space"},
            "click": {"type": "mouse_click", "button": "left"},
            "wait": {"type": "wait", "duration": 0.15},
        }

    def get_objective(self) -> str:
        return "Stay alive. Fly through gaps between pipes. Flap to go up, do nothing to fall."