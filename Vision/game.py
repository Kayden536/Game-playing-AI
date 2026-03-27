import ollama
import pyautogui
import io
import json
import time
import sys
from PIL import ImageGrab, ImageEnhance, Image

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


# ============================================================
#  VISION — simplified and faster
# ============================================================

class Vision:
    def __init__(self, model="gemma3"):
        self.model = model
        self.screen_size = pyautogui.size()
        print(f"👁️  Vision: {model}")

    def capture_screen(self, region=None):
        screenshot = ImageGrab.grab(bbox=region)
        self.original_size = screenshot.size

        # SMALLER = FASTER
        max_dim = 672
        w, h = screenshot.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            screenshot = screenshot.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

        screenshot = ImageEnhance.Contrast(screenshot).enhance(1.3)

        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    def analyze(self, region=None, game_context=""):
        image_bytes = self.capture_screen(region)

        prompt = f"""Look at this game screen. {game_context}

Reply with ONLY this short JSON:

{{
    "state": "playing or menu or game_over",
    "player_y": 0-100,
    "nearest_obstacle_x": 0-100,
    "gap_y": 0-100,
    "action_needed": "what to do right now in 5 words max"
}}

JSON only. No other text."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }],
                options={
                    "temperature": 0.1,
                    "num_predict": 200,  # SHORT response
                },
            )

            raw = response["message"]["content"]
            print(f"👁️  Raw: {raw[:150]}")
            return self._parse_json(raw)

        except Exception as e:
            print(f"👁️  Error: {e}")
            return None

    def _parse_json(self, raw_text):
        raw_text = raw_text.strip()
        # Strip markdown
        for prefix in ["```json", "```"]:
            if raw_text.startswith(prefix):
                raw_text = raw_text[len(prefix):]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw_text[start:end])
            except json.JSONDecodeError:
                pass

        return {"state": "unknown", "action_needed": "unknown", "raw": raw_text}


# ============================================================
#  BRAIN — simplified and faster
# ============================================================

class Brain:
    def __init__(self, model="mistral"):
        self.model = model
        print(f"🧠 Brain: {model}")

    def decide(self, game_state, game_rules=""):
        if game_state is None:
            return {"action": "wait", "reason": "no vision data"}

        prompt = f"""You are playing a game.

WHAT YOU SEE:
{json.dumps(game_state)}

RULES:
{game_rules}

What ONE action to take? Reply with ONLY this JSON:

{{"action": "press_key or click or wait", "key": "space", "reason": "why"}}

JSON only."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with JSON only. No other text ever.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 150,  # SHORT
                },
            )

            raw = response["message"]["content"]
            print(f"🧠 Raw: {raw[:150]}")
            return self._parse_json(raw)

        except Exception as e:
            print(f"🧠 Error: {e}")
            return {"action": "press_key", "key": "space", "reason": "error fallback"}

    def _parse_json(self, raw_text):
        raw_text = raw_text.strip()
        for prefix in ["```json", "```"]:
            if raw_text.startswith(prefix):
                raw_text = raw_text[len(prefix):]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw_text[start:end])
            except json.JSONDecodeError:
                pass

        # FALLBACK — just press space
        print("⚠️  Parse failed, fallback to space")
        return {"action": "press_key", "key": "space", "reason": "parse fallback"}


# ============================================================
#  ACTIONS — same but with better logging
# ============================================================

class Actions:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.region = None
        print(f"🎯 Actions: {self.screen_w}x{self.screen_h}")

    def set_game_region(self, x, y, w, h):
        self.region = (x, y, w, h)

    def percent_to_screen(self, x_pct, y_pct):
        if self.region:
            rx, ry, rw, rh = self.region
            return rx + int((x_pct / 100) * rw), ry + int((y_pct / 100) * rh)
        return int((x_pct / 100) * self.screen_w), int((y_pct / 100) * self.screen_h)

    def execute(self, action_data):
        action = action_data.get("action", "wait")
        reason = action_data.get("reason", "")

        print(f"   ▶ {action.upper()} | {reason}")

        try:
            if action == "click":
                x, y = self.percent_to_screen(
                    action_data.get("x", action_data.get("x_percent", 50)),
                    action_data.get("y", action_data.get("y_percent", 50)),
                )
                print(f"     clicking ({x}, {y})")
                pyautogui.click(x, y)

            elif action == "press_key":
                key = action_data.get("key", "space")
                print(f"     pressing '{key}'")
                pyautogui.press(key)

            elif action == "hold_key":
                key = action_data.get("key", "space")
                dur = action_data.get("duration", 0.1)
                pyautogui.keyDown(key)
                time.sleep(dur)
                pyautogui.keyUp(key)

            elif action == "wait":
                time.sleep(0.3)

            else:
                print(f"     unknown action: {action}")
                # Fallback: press space
                pyautogui.press("space")

            return True

        except Exception as e:
            print(f"     ❌ {e}")
            return False


# ============================================================
#  GAME LOOP
# ============================================================

class GameAI:
    def __init__(self, vision_model="gemma3", brain_model="mistral",
                 game_region=None):
        print("=" * 40)
        print("  🎮 AI Game Player (Fast)")
        print("=" * 40)

        self.vision = Vision(model=vision_model)
        self.brain = Brain(model=brain_model)
        self.actions = Actions()

        if game_region:
            self.actions.set_game_region(*game_region)

        self.game_rules = ""
        self.loop_count = 0

    def set_game_rules(self, rules):
        self.game_rules = rules

    def step(self):
        self.loop_count += 1
        print(f"\n--- Loop #{self.loop_count} ---")

        # SEE
        t0 = time.time()
        game_state = self.vision.analyze(
            region=self.actions.region,
            game_context=self.game_rules,
        )
        print(f"👁️  Vision: {time.time() - t0:.1f}s")

        # THINK
        t0 = time.time()
        action = self.brain.decide(game_state, self.game_rules)
        print(f"🧠 Brain: {time.time() - t0:.1f}s")

        # ACT
        self.actions.execute(action)

        return game_state, action

    def run(self, max_loops=None, delay=0.5):
        print(f"\n🚀 Starting in 3...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)

        loop = 0
        try:
            while True:
                if max_loops and loop >= max_loops:
                    break

                self.step()
                time.sleep(delay)
                loop += 1

        except KeyboardInterrupt:
            print("\n⛔ Stopped")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe")

        print(f"📊 Done: {loop} loops")


# ============================================================
#  PRESETS
# ============================================================

PRESETS = {
    "flappy_bird": "click to flap. Avoid pipes. If bird is low, click. If bird is high, wait. If Game Over click X=800 and Y=215",
    "cookie_clicker": "Click the big cookie. Buy cheapest upgrade when possible.",
    "dino_run": "Press SPACE to jump over cacti. Press DOWN to duck under birds.",
    "2048": "Use arrow keys. Keep highest tile in corner.",
}


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n🎮 Setup\n")

    print("Games:")
    presets = list(PRESETS.keys())
    for i, name in enumerate(presets, 1):
        print(f"  {i}. {name}")
    print(f"  {len(presets) + 1}. custom")

    choice = input("\nPick: ").strip()
    rules = ""
    if choice.isdigit() and int(choice) <= len(presets):
        rules = PRESETS[presets[int(choice) - 1]]
        print(f"✅ {presets[int(choice) - 1]}")
    else:
        rules = input("Game rules: ").strip()

    v_model = input("Vision model [gemma3]: ").strip() or "gemma3"
    b_model = input("Brain model [mistral]: ").strip() or "mistral"
    delay = float(input("Loop delay seconds [0.5]: ").strip() or "0.5")
    max_l = input("Max loops [empty=unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None

    region_input = input("Game region x,y,w,h [empty=fullscreen]: ").strip()
    region = None
    if region_input:
        try:
            region = tuple(int(p) for p in region_input.split(","))
        except ValueError:
            pass

    ai = GameAI(vision_model=v_model, brain_model=b_model, game_region=region)
    ai.set_game_rules(rules)

    mode = input("(r)un or (s)tep? [r]: ").strip().lower()
    if mode == "s":
        while True:
            ai.step()
            if input("\nNext? (Enter/q): ").strip().lower() == "q":
                break
    else:
        ai.run(max_loops=max_loops, delay=delay)


if __name__ == "__main__":
    main()
    