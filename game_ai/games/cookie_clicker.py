from games.base_game import BaseGame


class CookieClickerProfile(BaseGame):
    def __init__(self):
        super().__init__()
        self.name = "Cookie Clicker"
        self.game_specific_context = """
        This is Cookie Clicker, a browser incremental game.
        - Large cookie on the left side - click it to earn cookies
        - Right side has upgrades and buildings to purchase
        - Buy cheapest available upgrade/building when possible
        - Golden cookies appear randomly - click them immediately
        - Cookie count shown at the top
        - More buildings = more cookies per second
        """
        # Cookie Clicker is typically in a browser window
        self.screen_region = {
            "left": 0, "top": 0,
            "width": 1920, "height": 1080
        }

    def get_action_map(self) -> dict:
        return {
            "click": {"type": "mouse_click", "button": "left"},
            "wait": {"type": "wait", "duration": 0.1},
        }

    def get_objective(self) -> str:
        return "Click the big cookie. Buy upgrades and buildings when available. Click golden cookies."