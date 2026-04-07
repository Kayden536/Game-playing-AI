import ollama
import pyautogui
import io
import json
import time
import sys
from PIL import ImageGrab, ImageEnhance, Image, ImageDraw, ImageFont

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05
 

# ============================================================
#  VISION — with button detection and grid overlay
# ============================================================

class Vision:
    def __init__(self, model="gemma3"):
        self.model = model
        self.screen_size = pyautogui.size()
        self.original_size = None
        self.scaled_size = None
        print(f"👁️  Vision: {model}")

    def capture_screen(self, region=None, add_grid=False):
        """Capture screen, optionally with a coordinate grid overlay."""
        screenshot = ImageGrab.grab(bbox=region)
        self.original_size = screenshot.size

        # Resize for speed
        max_dim = 768
        w, h = screenshot.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            screenshot = screenshot.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        self.scaled_size = screenshot.size

        screenshot = ImageEnhance.Contrast(screenshot).enhance(1.3)

        # Draw grid overlay so the model can estimate positions
        if add_grid:
            screenshot = self._draw_grid(screenshot)

        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

    def _draw_grid(self, img):
        """Draw a labeled percentage grid on the image to help
        the model report accurate coordinates."""
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Try to load a small font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        # Vertical lines every 10 %
        for pct in range(0, 101, 10):
            x = int(pct / 100 * w)
            draw.line([(x, 0), (x, h)], fill=(255, 0, 0, 80), width=1)
            draw.text((x + 2, 2), f"{pct}%", fill=(255, 0, 0), font=font)

        # Horizontal lines every 10 %
        for pct in range(0, 101, 10):
            y = int(pct / 100 * h)
            draw.line([(0, y), (w, y)], fill=(0, 0, 255, 80), width=1)
            draw.text((2, y + 2), f"{pct}%", fill=(0, 0, 255), font=font)

        return img

    # ----------------------------------------------------------
    #  Primary game-state analysis (used every loop)
    # ----------------------------------------------------------
    def analyze(self, region=None, game_context=""):
        image_bytes = self.capture_screen(region, add_grid=True)

        prompt = f"""You are an AI controlling a game.
Look at this screenshot. Red vertical lines = x%. Blue horizontal lines = y%.
Use those grid lines to estimate positions accurately.

{game_context}

Reply with ONLY this JSON (no other text):

{{
    "state": "playing" or "menu" or "game_over" or "paused",
    "buttons": [
        {{
            "label": "human-readable label like Play, Restart, OK",
            "x_percent": 50,
            "y_percent": 50
        }}
    ],
    "player_y": 0-100,
    "nearest_obstacle_x": 0-100,
    "gap_y": 0-100,
    "action": "press_key" or "click" or "move_click" or "wait",
    "key": "space",
    "click_x": 50,
    "click_y": 50,
    "reason": "brief reason"
}}

RULES:
- "buttons" = every clickable button/link you can see with its center
  coordinates as x_percent (0-100) and y_percent (0-100).
- If state is "menu" or "game_over", find the Play/Restart/Retry
  button and set action="move_click" with click_x/click_y pointing
  at that button's center.
- If state is "playing", decide whether to press a key or wait.
- Use the red/blue grid lines to estimate coordinates precisely.

JSON only. No markdown fences. No explanation."""

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
                    "num_predict": 400,
                },
            )

            raw = response["message"]["content"]
            print(f"👁️  Raw: {raw[:200]}")
            return self._parse_json(raw)

        except Exception as e:
            print(f"👁️  Error: {e}")
            return None

    # ----------------------------------------------------------
    #  Dedicated button finder (called when we need to locate
    #  a specific button like Retry / Play / Start)
    # ----------------------------------------------------------
    def find_buttons(self, region=None, button_hint=""):
        """Ask the model specifically to find all clickable buttons."""
        image_bytes = self.capture_screen(region, add_grid=True)

        prompt = f"""Look at this screenshot with a percentage grid overlay.
Red vertical lines show x-percentages, blue horizontal lines show y-percentages.

Find ALL clickable buttons, links, or interactive elements.
{f'I am especially looking for: {button_hint}' if button_hint else ''}

Reply with ONLY this JSON array (no other text):

[
    {{
        "label": "button text or description",
        "x_percent": 50,
        "y_percent": 50,
        "confidence": "high" or "medium" or "low"
    }}
]

Use the grid lines to estimate the CENTER of each button precisely.
JSON array only. No markdown. No explanation."""

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
                    "num_predict": 500,
                },
            )

            raw = response["message"]["content"]
            print(f"🔍 Buttons raw: {raw[:250]}")
            return self._parse_json(raw)

        except Exception as e:
            print(f"🔍 Button scan error: {e}")
            return []

    # ----------------------------------------------------------
    #  Verify a specific screen location (double-check before
    #  clicking something important)
    # ----------------------------------------------------------
    def verify_location(self, x_pct, y_pct, region=None, expected="button"):
        """Take a second look at a specific area to confirm there's
        something clickable there."""
        image_bytes = self.capture_screen(region, add_grid=True)

        prompt = f"""I am about to click at x={x_pct}%, y={y_pct}% on this screenshot.
The grid lines show percentages.

Is there a clickable {expected} near that location?

Reply with ONLY this JSON:
{{
    "correct": true or false,
    "adjusted_x": {x_pct},
    "adjusted_y": {y_pct},
    "what_is_there": "description"
}}

If the location is slightly off, provide corrected coordinates.
JSON only."""

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
                    "num_predict": 150,
                },
            )

            raw = response["message"]["content"]
            print(f"✅ Verify: {raw[:150]}")
            return self._parse_json(raw)

        except Exception as e:
            print(f"✅ Verify error: {e}")
            return {"correct": True, "adjusted_x": x_pct, "adjusted_y": y_pct}

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

        # Try finding outermost JSON object or array
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = raw_text.find(open_ch)
            end = raw_text.rfind(close_ch) + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw_text[start:end])
                except json.JSONDecodeError:
                    continue

        return {"state": "unknown", "raw": raw_text}


# ============================================================
#  BRAIN — optional separate reasoning model
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
{json.dumps(game_state, indent=2)}

RULES:
{game_rules}

IMPORTANT:
- If state is "menu" or "game_over" and buttons are listed,
  pick the best button (Play/Restart/Retry) and use action="move_click"
  with that button's click_x and click_y.
- If state is "playing", decide press_key or wait.

Reply with ONLY this JSON:

{{"action": "press_key or move_click or wait", "key": "space", "click_x": 50, "click_y": 50, "reason": "why"}}

JSON only."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Reply with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": 200},
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
        return {"action": "press_key", "key": "space", "reason": "parse fallback"}


# ============================================================
#  ACTIONS — full cursor control
# ============================================================

class Actions:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.region = None
        print(f"🎯 Actions: screen {self.screen_w}x{self.screen_h}")

    def set_game_region(self, x, y, w, h):
        self.region = (x, y, w, h)
        print(f"🎯 Game region: {self.region}")

    def percent_to_screen(self, x_pct, y_pct):
        """Convert 0-100 percentages to absolute screen coordinates,
        accounting for game region if set."""
        if self.region:
            rx, ry, rw, rh = self.region
            return (
                rx + int((x_pct / 100) * rw),
                ry + int((y_pct / 100) * rh),
            )
        return (
            int((x_pct / 100) * self.screen_w),
            int((y_pct / 100) * self.screen_h),
        )

    def move_cursor(self, x, y, duration=0.3):
        """Smoothly move cursor to absolute screen coordinates."""
        print(f"     🖱️  moving cursor → ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)

    def move_and_click(self, x_pct, y_pct, duration=0.3):
        """Move cursor to percentage position, pause, then click."""
        x, y = self.percent_to_screen(x_pct, y_pct)
        print(f"     🖱️  move_click: {x_pct}%,{y_pct}% → ({x},{y})")
        pyautogui.moveTo(x, y, duration=duration)
        time.sleep(0.15)          # brief pause so the game registers hover
        pyautogui.click(x, y)
        return x, y

    def execute(self, action_data):
        action = action_data.get("action", "wait")
        reason = action_data.get("reason", "")
        print(f"   ▶ {action.upper()} | {reason}")

        try:
            # ---- MOVE + CLICK (primary way to click buttons) ----
            if action in ("move_click", "click"):
                x_pct = action_data.get(
                    "click_x",
                    action_data.get("x_percent", action_data.get("x", 50)),
                )
                y_pct = action_data.get(
                    "click_y",
                    action_data.get("y_percent", action_data.get("y", 50)),
                )
                self.move_and_click(x_pct, y_pct)

            # ---- PRESS KEY ----
            elif action == "press_key":
                key = action_data.get("key", "space")
                print(f"     ⌨️  pressing '{key}'")
                pyautogui.press(key)

            # ---- HOLD KEY ----
            elif action == "hold_key":
                key = action_data.get("key", "space")
                dur = action_data.get("duration", 0.15)
                print(f"     ⌨️  holding '{key}' for {dur}s")
                pyautogui.keyDown(key)
                time.sleep(dur)
                pyautogui.keyUp(key)

            # ---- MOVE ONLY (no click) ----
            elif action == "move":
                x_pct = action_data.get("click_x", 50)
                y_pct = action_data.get("click_y", 50)
                x, y = self.percent_to_screen(x_pct, y_pct)
                self.move_cursor(x, y)

            # ---- WAIT ----
            elif action == "wait":
                time.sleep(0.3)

            else:
                print(f"     ⚠️  unknown action '{action}', pressing space")
                pyautogui.press("space")

            return True

        except Exception as e:
            print(f"     ❌ {e}")
            return False


# ============================================================
#  GAME AI — with state machine for menus / game-over
# ============================================================

# Keywords the AI might use to label restart-type buttons
RESTART_KEYWORDS = [
    "restart", "retry", "play", "start", "again",
    "continue", "new game", "replay", "ok", "tap",
]


def _pick_best_button(buttons, keywords=None):
    """From a list of button dicts, pick the one whose label best
    matches common restart/play keywords."""
    if not buttons:
        return None
    if keywords is None:
        keywords = RESTART_KEYWORDS

    # Prefer high-confidence matches
    for confidence in ("high", "medium", "low"):
        for btn in buttons:
            if btn.get("confidence", "medium") != confidence:
                continue
            label = btn.get("label", "").lower()
            for kw in keywords:
                if kw in label:
                    return btn

    # Fallback: first button
    return buttons[0] if buttons else None


class GameAI:
    def __init__(
        self,
        vision_model="gemma3",
        brain_model="mistral",
        game_region=None,
        vision_only=True,
        verify_clicks=False,
    ):
        print("=" * 50)
        print("  🎮  AI Game Player — Autonomous Cursor Control")
        print("=" * 50)

        self.vision = Vision(model=vision_model)
        self.vision_only = vision_only
        self.brain = None if vision_only else Brain(model=brain_model)
        self.actions = Actions()
        self.verify_clicks = verify_clicks

        if game_region:
            self.actions.set_game_region(*game_region)

        self.game_rules = ""
        self.loop_count = 0
        self.last_state = None
        self.consecutive_game_over = 0     # track repeated game-overs

        mode = "FAST (vision-only)" if vision_only else "FULL (vision + brain)"
        print(f"⚡ Mode: {mode}")
        print(f"🔍 Click verification: {'ON' if verify_clicks else 'OFF'}")

    def set_game_rules(self, rules):
        self.game_rules = rules

    # ----------------------------------------------------------
    #  Handle non-playing states (menu, game_over, paused)
    # ----------------------------------------------------------
    def _handle_non_playing(self, game_state):
        """When not actively playing, find and click the right button."""
        state = game_state.get("state", "unknown")
        print(f"   📋 State: {state} — scanning for buttons …")

        # 1. Check if the vision analysis already found buttons
        buttons = game_state.get("buttons", [])

        # 2. If no buttons found, do a dedicated button scan
        if not buttons:
            hint = "Retry, Restart, or Play button"
            buttons = self.vision.find_buttons(
                region=self.actions.region,
                button_hint=hint,
            )
            # find_buttons might return a dict on error
            if isinstance(buttons, dict):
                buttons = buttons.get("buttons", [])

        if not buttons:
            print("   ⚠️  No buttons found — pressing space as fallback")
            return {"action": "press_key", "key": "space", "reason": "no buttons found"}

        print(f"   🔘 Found {len(buttons)} button(s):")
        for b in buttons:
            print(f"      • {b.get('label','?')} @ "
                  f"({b.get('x_percent', '?')}%, {b.get('y_percent', '?')}%) "
                  f"[{b.get('confidence', '?')}]")

        # 3. Pick the best button
        best = _pick_best_button(buttons)
        if not best:
            print("   ⚠️  Could not pick a button")
            return {"action": "press_key", "key": "space", "reason": "no match"}

        x_pct = best.get("x_percent", 50)
        y_pct = best.get("y_percent", 50)

        # 4. Optionally verify the location
        if self.verify_clicks:
            verify = self.vision.verify_location(
                x_pct, y_pct,
                region=self.actions.region,
                expected=best.get("label", "button"),
            )
            if isinstance(verify, dict):
                x_pct = verify.get("adjusted_x", x_pct)
                y_pct = verify.get("adjusted_y", y_pct)
                print(f"   ✅ Verified → ({x_pct}%, {y_pct}%): "
                      f"{verify.get('what_is_there', '?')}")

        return {
            "action": "move_click",
            "click_x": x_pct,
            "click_y": y_pct,
            "reason": f"clicking '{best.get('label', 'button')}'",
        }

    # ----------------------------------------------------------
    #  One game loop iteration
    # ----------------------------------------------------------
    def step(self):
        self.loop_count += 1
        print(f"\n{'='*40}  Loop #{self.loop_count}  {'='*40}")

        # ---- SEE ----
        t0 = time.time()
        game_state = self.vision.analyze(
            region=self.actions.region,
            game_context=self.game_rules,
        )
        vision_time = time.time() - t0
        print(f"👁️  Vision took {vision_time:.1f}s")

        if game_state is None:
            print("👁️  No data — waiting")
            time.sleep(0.5)
            return None, {"action": "wait"}

        state = game_state.get("state", "unknown")

        # ---- DECIDE ----
        if state in ("menu", "game_over", "paused"):
            self.consecutive_game_over += 1
            action = self._handle_non_playing(game_state)

            # If stuck game-over for many loops, try clicking center
            if self.consecutive_game_over > 5:
                print("   ⚠️  Stuck — trying center click")
                action = {
                    "action": "move_click",
                    "click_x": 50,
                    "click_y": 50,
                    "reason": "stuck fallback center click",
                }
                self.consecutive_game_over = 0

        elif state == "playing":
            self.consecutive_game_over = 0

            if self.vision_only:
                # Use vision's own action recommendation
                action = game_state
            else:
                t0 = time.time()
                action = self.brain.decide(game_state, self.game_rules)
                print(f"🧠 Brain took {time.time() - t0:.1f}s")
        else:
            # Unknown state — try what vision suggests
            action = game_state

        # ---- ACT ----
        self.actions.execute(action)

        self.last_state = game_state
        return game_state, action

    # ----------------------------------------------------------
    #  Run loop
    # ----------------------------------------------------------
    def run(self, max_loops=None, delay=0.5):
        print(f"\n🚀 Starting in 3 …")
        for i in range(3, 0, -1):
            print(f"   {i} …")
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
            print("\n⛔ Stopped by user")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe triggered (mouse in corner)")

        print(f"\n📊 Finished after {loop} loops")


# ============================================================
#  PRESETS
# ============================================================

PRESETS = {
    "flappy_bird": (
        "Flappy Bird: press_key with key='space' to flap. "
        "If bird low or obstacle close, press space. If bird high, wait. "
        "On game-over, find the Retry/Restart button and move_click it."
    ),
    "cookie_clicker": "Click the big cookie. Buy cheapest upgrade when possible.",
    "dino_run": "Press SPACE to jump over cacti. Press DOWN to duck.",
    "2048": "Use arrow keys (up/down/left/right). Keep highest tile in corner.",
}


# ============================================================
#  INTERACTIVE SETUP
# ============================================================

def _region_picker():
    """Let the user click two corners to define the game region."""
    print("\n🖱️  Region picker:")
    print("   Move your mouse to the TOP-LEFT corner of the game and press Enter")
    input("   → ")
    x1, y1 = pyautogui.position()
    print(f"   Got ({x1}, {y1})")

    print("   Now move to the BOTTOM-RIGHT corner and press Enter")
    input("   → ")
    x2, y2 = pyautogui.position()
    print(f"   Got ({x2}, {y2})")

    region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    print(f"   ✅ Region: {region}")
    return region


def main():
    print("\n🎮 AI Game Player — Setup\n")

    # ---- Game preset ----
    presets = list(PRESETS.keys())
    for i, name in enumerate(presets, 1):
        print(f"  {i}. {name}")
    print(f"  {len(presets) + 1}. custom")

    choice = input("\nPick game: ").strip()
    rules = ""
    if choice.isdigit() and int(choice) <= len(presets):
        rules = PRESETS[presets[int(choice) - 1]]
        print(f"✅ {presets[int(choice) - 1]}")
    else:
        rules = input("Describe game rules: ").strip()

    # ---- Models ----
    v_model = input("Vision model [gemma3]: ").strip() or "gemma3"
    use_brain = input("Separate brain model? (y/n) [n]: ").strip().lower() == "y"
    b_model = "mistral"
    if use_brain:
        b_model = input("Brain model [mistral]: ").strip() or "mistral"

    # ---- Timing ----
    delay = float(input("Loop delay seconds [0.3]: ").strip() or "0.3")
    max_l = input("Max loops [empty=unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None

    # ---- Region ----
    region = None
    reg_choice = input("Game region — (m)anual coords, (p)ick with mouse, (f)ullscreen [f]: ").strip().lower()
    if reg_choice == "m":
        coords = input("  x,y,w,h: ").strip()
        try:
            region = tuple(int(p) for p in coords.split(","))
        except ValueError:
            print("  ⚠️  Bad format, using fullscreen")
    elif reg_choice == "p":
        region = _region_picker()

    # ---- Verification ----
    verify = input("Verify clicks before executing? (y/n) [n]: ").strip().lower() == "y"

    # ---- Build & Run ----
    ai = GameAI(
        vision_model=v_model,
        brain_model=b_model,
        game_region=region,
        vision_only=not use_brain,
        verify_clicks=verify,
    )
    ai.set_game_rules(rules)

    mode = input("\n(r)un continuously or (s)tep-by-step? [r]: ").strip().lower()
    if mode == "s":
        while True:
            ai.step()
            if input("\nNext? (Enter/q): ").strip().lower() == "q":
                break
    else:
        ai.run(max_loops=max_loops, delay=delay)


if __name__ == "__main__":
    main()