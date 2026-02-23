"""
Universal Game-Playing AI
Usage: python main.py --game minecraft
"""

import argparse
import os

# Create logs directory
os.makedirs("logs", exist_ok=True)


def get_game_profile(game_name: str):
    """Load game profile by name."""
    games = {
        "minecraft": "games.minecraft.MinecraftProfile",
        "flappy_bird": "games.flappy_bird.FlappyBirdProfile",
        "flappy": "games.flappy_bird.FlappyBirdProfile",
        "cookie_clicker": "games.cookie_clicker.CookieClickerProfile",
        "cookie": "games.cookie_clicker.CookieClickerProfile",
    }

    if game_name.lower() not in games:
        print(f"Unknown game: {game_name}")
        print(f"Available games: {list(games.keys())}")
        print(f"\nYou can also create a custom profile!")
        return None

    # Dynamic import
    module_path, class_name = games[game_name.lower()].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    profile_class = getattr(module, class_name)
    return profile_class()


def create_custom_game():
    """Interactive custom game profile creation."""
    from games.base_game import BaseGame

    print("\n=== Custom Game Setup ===")
    name = input("Game name: ")
    objective = input("Objective (what should the AI try to do?): ")
    context = input("Game description (help the AI understand the game): ")

    print("\nWhat inputs does this game use?")
    print("Common options: wasd, arrow_keys, space, click, mouse_move")
    inputs = input("Controls (comma-separated): ").split(",")

    class CustomGame(BaseGame):
        def __init__(self):
            super().__init__()
            self.name = name
            self.game_specific_context = context

        def get_action_map(self):
            action_map = {"wait": {"type": "wait", "duration": 0.3}}
            input_mappings = {
                "wasd": {
                    "move_forward": {"type": "key_hold", "key": "w"},
                    "move_backward": {"type": "key_hold", "key": "s"},
                    "move_left": {"type": "key_hold", "key": "a"},
                    "move_right": {"type": "key_hold", "key": "d"},
                },
                "arrow_keys": {
                    "up": {"type": "key_press", "key": "up"},
                    "down": {"type": "key_press", "key": "down"},
                    "left": {"type": "key_press", "key": "left"},
                    "right": {"type": "key_press", "key": "right"},
                },
                "space": {
                    "jump": {"type": "key_press", "key": "space"},
                },
                "click": {
                    "click": {"type": "mouse_click", "button": "left"},
                    "right_click": {"type": "mouse_click", "button": "right"},
                },
                "mouse_move": {
                    "look_left": {"type": "mouse_move", "x": -100, "y": 0},
                    "look_right": {"type": "mouse_move", "x": 100, "y": 0},
                    "look_up": {"type": "mouse_move", "x": 0, "y": -50},
                    "look_down": {"type": "mouse_move", "x": 0, "y": 50},
                },
            }
            for inp in inputs:
                inp = inp.strip().lower()
                if inp in input_mappings:
                    action_map.update(input_mappings[inp])
            return action_map

        def get_objective(self):
            return objective

    return CustomGame()


def main():
    parser = argparse.ArgumentParser(description="Universal Game-Playing AI")
    parser.add_argument(
        "--game", type=str, default=None,
        help="Game to play (minecraft, flappy_bird, cookie_clicker)"
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Create a custom game profile"
    )
    parser.add_argument(
        "--countdown", type=int, default=5,
        help="Seconds before AI starts"
    )
    args = parser.parse_args()

    # Get or create game profile
    if args.custom:
        game_profile = create_custom_game()
    elif args.game:
        game_profile = get_game_profile(args.game)
        if game_profile is None:
            return
    else:
        print("Universal Game-Playing AI")
        print("=" * 40)
        print("\nUsage:")
        print("  python main.py --game minecraft")
        print("  python main.py --game flappy_bird")
        print("  python main.py --game cookie_clicker")
        print("  python main.py --custom")
        print("\nAvailable games: minecraft, flappy_bird, cookie_clicker")
        return

    # Create and start agent
    from core.agent import GameAgent
    agent = GameAgent(game_profile)
    agent.start(countdown=args.countdown)


if __name__ == "__main__":
    main()