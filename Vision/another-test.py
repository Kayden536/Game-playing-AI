import ollama
import pyautogui
import io
import json
import time
import os
from PIL import ImageGrab, ImageEnhance, Image, ImageDraw, ImageFont

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

 
# ============================================================
#  VISION — numbered grid + two-pass refinement
# ============================================================

class Vision:
    def __init__(self, model="gemma3", debug=True):
        self.model = model
        self.debug = debug
        # Coarse grid: 5 columns × 4 rows = 20 cells
        self.grid_cols = 5
        self.grid_rows = 4
        # Fine grid for refinement: 3×3 = 9 sub-cells
        self.refine_cols = 3
        self.refine_rows = 3
        self.frame = 0

        if debug:
            os.makedirs("debug_frames", exist_ok=True)
        print(f"👁️  Vision: {model} | debug={'ON' if debug else 'OFF'}")

    # ---------- capture ----------

    def capture(self, region=None):
        """region = (x, y, w, h) or None for fullscreen.
        Returns a PIL Image (resized for the model)."""
        if region:
            x, y, w, h = region
            # *** FIX: ImageGrab wants (left, top, RIGHT, BOTTOM) ***
            bbox = (x, y, x + w, y + h)
        else:
            bbox = None

        shot = ImageGrab.grab(bbox=bbox)

        # Resize — keep aspect ratio
        max_dim = 768
        sw, sh = shot.size
        if max(sw, sh) > max_dim:
            scale = max_dim / max(sw, sh)
            shot = shot.resize(
                (int(sw * scale), int(sh * scale)), Image.LANCZOS
            )

        shot = ImageEnhance.Contrast(shot).enhance(1.2)
        return shot

    # ---------- grid drawing ----------

    def _draw_grid(self, img, rows, cols):
        """Draw numbered cells. Numbers are large and centered."""
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cw = w / cols
        ch = h / rows

        try:
            fsize = max(16, int(min(cw, ch) * 0.35))
            font = ImageFont.truetype("arial.ttf", fsize)
        except Exception:
            font = ImageFont.load_default()
            fsize = 10

        num = 1
        for r in range(rows):
            for c in range(cols):
                x1 = int(c * cw)
                y1 = int(r * ch)
                x2 = int((c + 1) * cw)
                y2 = int((r + 1) * ch)

                # Red cell border
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

                # Centered number with black pill background
                label = str(num)
                tw = fsize * len(label) + 10
                th = fsize + 8
                cx = (x1 + x2) // 2 - tw // 2
                cy = (y1 + y2) // 2 - th // 2
                draw.rectangle(
                    [cx, cy, cx + tw, cy + th],
                    fill=(0, 0, 0),
                )
                draw.text(
                    (cx + 5, cy + 4), label,
                    fill=(255, 255, 0), font=font,
                )
                num += 1

        return img

    # ---------- cell math ----------

    def _cell_center_pct(self, cell_num, rows, cols):
        """1-based cell number → (x_percent, y_percent) of cell center."""
        idx = cell_num - 1
        r = idx // cols
        c = idx % cols
        return (c + 0.5) / cols * 100, (r + 0.5) / rows * 100

    def _crop_cell(self, img, cell_num, rows, cols):
        """Crop one cell out of the image."""
        idx = cell_num - 1
        r = idx // cols
        c = idx % cols
        w, h = img.size
        cw = w / cols
        ch = h / rows
        return img.crop((
            int(c * cw), int(r * ch),
            int((c + 1) * cw), int((r + 1) * ch),
        ))

    def _refine_pct(self, coarse_cell, fine_cell, coarse_rows, coarse_cols,
                    fine_rows, fine_cols):
        """Combine coarse cell + fine sub-cell into final percent coords."""
        # Coarse cell boundaries in percent
        cidx = coarse_cell - 1
        cr = cidx // coarse_cols
        cc = cidx % coarse_cols
        x1 = cc / coarse_cols * 100
        y1 = cr / coarse_rows * 100
        cell_w = 100 / coarse_cols
        cell_h = 100 / coarse_rows

        # Fine sub-cell center within that coarse cell
        fidx = fine_cell - 1
        fr = fidx // fine_cols
        fc = fidx % fine_cols
        fx = (fc + 0.5) / fine_cols
        fy = (fr + 0.5) / fine_rows

        return x1 + fx * cell_w, y1 + fy * cell_h

    # ---------- helpers ----------

    def _to_bytes(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _save_debug(self, img, tag):
        if self.debug:
            self.frame += 1
            path = f"debug_frames/{self.frame:04d}_{tag}.png"
            img.save(path)
            print(f"   📸 {path}")

    def _ask(self, prompt, image_bytes, max_tokens=300):
        """Send a prompt + image to the model, return raw text."""
        resp = ollama.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }],
            options={"temperature": 0.1, "num_predict": max_tokens},
        )
        return resp["message"]["content"]

    def _parse_json(self, raw):
        raw = raw.strip()
        for p in ("```json", "```"):
            if raw.startswith(p):
                raw = raw[len(p):]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        for o, c in [("{", "}"), ("[", "]")]:
            s = raw.find(o)
            e = raw.rfind(c) + 1
            if s != -1 and e > s:
                try:
                    return json.loads(raw[s:e])
                except json.JSONDecodeError:
                    continue
        return None

    # ==========================================================
    #  MAIN ANALYSIS — returns game state + action
    # ==========================================================

    def analyze(self, region=None, game_context=""):
        base_img = self.capture(region)
        grid_img = self._draw_grid(base_img, self.grid_rows, self.grid_cols)
        self._save_debug(grid_img, "grid")

        total_cells = self.grid_rows * self.grid_cols
        prompt = f"""You are an AI playing a game. The screenshot has a numbered grid
(cells 1–{total_cells}, left→right, top→bottom, {self.grid_cols} columns × {self.grid_rows} rows).

{game_context}

Look at the screenshot and reply with ONLY this JSON:

{{
    "state": "playing" or "menu" or "game_over" or "paused",
    "player_cell": 0,
    "obstacle_cell": 0,
    "gap_cell": 0,
    "buttons": [
        {{"label": "button text", "cell": 7}}
    ],
    "action": "press_key" or "click_cell" or "wait",
    "target_cell": 0,
    "key": "space",
    "reason": "short reason"
}}

RULES:
- "buttons": list every button/link you see. "cell" = the grid cell
  number where its CENTER is.
- If state is "menu"/"game_over", find Play/Restart/Retry button,
  set action="click_cell" and target_cell to that button's cell number.
- If state is "playing", decide press_key or wait.
- target_cell is the cell number to click (1–{total_cells}).
- JSON only. No explanation."""

        raw = self._ask(prompt, self._to_bytes(grid_img), max_tokens=400)
        print(f"👁️  Raw: {raw[:200]}")
        result = self._parse_json(raw)

        if result is None:
            return {"state": "unknown", "action": "wait", "reason": "parse fail"}

        # If model wants to click a cell, convert to coordinates
        if result.get("action") == "click_cell" and result.get("target_cell"):
            cell = int(result["target_cell"])
            if 1 <= cell <= total_cells:
                # ---- REFINE: zoom into that cell and ask again ----
                x_pct, y_pct = self._refine_click(
                    base_img, cell, result.get("reason", "button")
                )
                result["click_x"] = x_pct
                result["click_y"] = y_pct
                result["action"] = "move_click"
                print(f"   🎯 Final coords: ({x_pct:.1f}%, {y_pct:.1f}%)")
            else:
                # Bad cell number — use center of screen
                result["click_x"] = 50
                result["click_y"] = 50
                result["action"] = "move_click"

        return result

    # ==========================================================
    #  REFINE — crop cell, subdivide, ask again
    # ==========================================================

    def _refine_click(self, base_img, coarse_cell, hint=""):
        """Zoom into a coarse cell, draw a 3×3 sub-grid, ask which
        sub-cell contains the click target. Returns (x_pct, y_pct)
        in full-image coordinates."""

        cropped = self._crop_cell(
            base_img, coarse_cell, self.grid_rows, self.grid_cols
        )
        # Scale up so the model can see detail
        cw, ch = cropped.size
        scale = max(1, 400 // max(cw, ch, 1))
        if scale > 1:
            cropped = cropped.resize(
                (cw * scale, ch * scale), Image.LANCZOS
            )

        fine_img = self._draw_grid(cropped, self.refine_rows, self.refine_cols)
        self._save_debug(fine_img, f"refine_cell{coarse_cell}")

        sub_total = self.refine_rows * self.refine_cols
        prompt = f"""This is a zoomed-in section of a game screen.
It has a {self.refine_cols}×{self.refine_rows} grid (cells 1–{sub_total}).
I want to click: {hint}

Which cell number contains the EXACT CENTER of what I should click?

Reply with ONLY this JSON:
{{"sub_cell": 5, "what": "description of what is there"}}

JSON only."""

        raw = self._ask(prompt, self._to_bytes(fine_img), max_tokens=100)
        print(f"   🔍 Refine: {raw[:120]}")
        parsed = self._parse_json(raw)

        if parsed and "sub_cell" in parsed:
            sub = int(parsed["sub_cell"])
            if 1 <= sub <= sub_total:
                x, y = self._refine_pct(
                    coarse_cell, sub,
                    self.grid_rows, self.grid_cols,
                    self.refine_rows, self.refine_cols,
                )
                return x, y

        # Fallback: center of coarse cell
        return self._cell_center_pct(
            coarse_cell, self.grid_rows, self.grid_cols
        )

    # ==========================================================
    #  FIND BUTTONS — dedicated button scanner
    # ==========================================================

    def find_buttons(self, region=None, hint=""):
        base_img = self.capture(region)
        grid_img = self._draw_grid(base_img, self.grid_rows, self.grid_cols)
        self._save_debug(grid_img, "button_scan")

        total = self.grid_rows * self.grid_cols
        prompt = f"""This screenshot has a numbered grid (cells 1–{total}).
Find ALL clickable buttons, links, or interactive elements.
{f'Especially looking for: {hint}' if hint else ''}

Reply with ONLY this JSON array:
[
    {{"label": "button text", "cell": 7}}
]

"cell" = grid cell number where the button's center is.
JSON array only."""

        raw = self._ask(prompt, self._to_bytes(grid_img), max_tokens=300)
        print(f"🔍 Buttons: {raw[:200]}")
        parsed = self._parse_json(raw)

        if isinstance(parsed, list):
            return parsed, base_img
        if isinstance(parsed, dict) and "buttons" in parsed:
            return parsed["buttons"], base_img
        return [], base_img


# ============================================================
#  ACTIONS — cursor control with coordinate fixes
# ============================================================

class Actions:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.region = None  # (x, y, w, h)
        print(f"🎯 Screen: {self.screen_w}×{self.screen_h}")

    def set_game_region(self, x, y, w, h):
        self.region = (x, y, w, h)
        print(f"🎯 Game region: x={x} y={y} w={w} h={h}")
        print(f"   → screen area: ({x},{y}) to ({x+w},{y+h})")

    def pct_to_abs(self, x_pct, y_pct):
        """Convert 0-100 percent to absolute screen pixel coords."""
        if self.region:
            rx, ry, rw, rh = self.region
            ax = rx + int(x_pct / 100 * rw)
            ay = ry + int(y_pct / 100 * rh)
        else:
            ax = int(x_pct / 100 * self.screen_w)
            ay = int(y_pct / 100 * self.screen_h)
        return ax, ay

    def execute(self, action_data):
        action = action_data.get("action", "wait")
        reason = action_data.get("reason", "")
        print(f"   ▶ {action.upper()} | {reason}")

        try:
            if action == "move_click":
                x_pct = float(action_data.get("click_x", 50))
                y_pct = float(action_data.get("click_y", 50))
                ax, ay = self.pct_to_abs(x_pct, y_pct)
                print(f"     🖱️  ({x_pct:.1f}%, {y_pct:.1f}%) → pixel ({ax}, {ay})")

                # Clamp to screen
                ax = max(0, min(ax, self.screen_w - 1))
                ay = max(0, min(ay, self.screen_h - 1))

                pyautogui.moveTo(ax, ay, duration=0.35)
                time.sleep(0.1)
                pyautogui.click(ax, ay)

            elif action == "click":
                # Legacy — same as move_click
                x_pct = float(action_data.get("click_x",
                              action_data.get("x_percent",
                              action_data.get("x", 50))))
                y_pct = float(action_data.get("click_y",
                              action_data.get("y_percent",
                              action_data.get("y", 50))))
                ax, ay = self.pct_to_abs(x_pct, y_pct)
                print(f"     🖱️  click ({x_pct:.1f}%, {y_pct:.1f}%) → ({ax}, {ay})")
                pyautogui.moveTo(ax, ay, duration=0.3)
                time.sleep(0.1)
                pyautogui.click(ax, ay)

            elif action == "press_key":
                key = action_data.get("key", "space")
                print(f"     ⌨️  '{key}'")
                pyautogui.press(key)

            elif action == "hold_key":
                key = action_data.get("key", "space")
                dur = float(action_data.get("duration", 0.15))
                print(f"     ⌨️  hold '{key}' {dur}s")
                pyautogui.keyDown(key)
                time.sleep(dur)
                pyautogui.keyUp(key)

            elif action == "wait":
                time.sleep(0.3)

            else:
                print(f"     ⚠️  unknown '{action}' → space")
                pyautogui.press("space")

            return True

        except Exception as e:
            print(f"     ❌ {e}")
            return False


# ============================================================
#  BRAIN — optional
# ============================================================

class Brain:
    def __init__(self, model="mistral"):
        self.model = model
        print(f"🧠 Brain: {model}")

    def decide(self, game_state, game_rules=""):
        if not game_state:
            return {"action": "wait", "reason": "no data"}

        prompt = f"""Game state:
{json.dumps(game_state, indent=2)}

Rules: {game_rules}

If state is menu/game_over and buttons exist, pick the best one and use
action="move_click" with click_x and click_y from the state data.
If playing, use press_key or wait.

Reply ONLY JSON: {{"action":"...", "key":"space", "click_x":50, "click_y":50, "reason":"..."}}"""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "JSON only."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": 200},
            )
            raw = resp["message"]["content"]
            print(f"🧠 {raw[:150]}")
            return self._parse(raw)
        except Exception as e:
            print(f"🧠 Error: {e}")
            return {"action": "press_key", "key": "space", "reason": "error"}

    def _parse(self, raw):
        raw = raw.strip()
        for p in ("```json", "```"):
            if raw.startswith(p): raw = raw[len(p):]
        if raw.endswith("```"): raw = raw[:-3]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except Exception:
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s != -1 and e > s:
                try:
                    return json.loads(raw[s:e])
                except Exception:
                    pass
        return {"action": "press_key", "key": "space", "reason": "parse fail"}


# ============================================================
#  BUTTON MATCHING
# ============================================================

RESTART_KW = ["restart", "retry", "play", "start", "again",
              "continue", "new", "replay", "ok", "tap", "begin"]

def _best_button(buttons):
    if not buttons:
        return None
    for btn in buttons:
        label = btn.get("label", "").lower()
        for kw in RESTART_KW:
            if kw in label:
                return btn
    return buttons[0]


# ============================================================
#  GAME AI
# ============================================================

class GameAI:
    def __init__(self, vision_model="gemma3", brain_model="mistral",
                 game_region=None, vision_only=True, debug=True):
        print("=" * 55)
        print("  🎮  AI Game Player — Numbered Grid + Refinement")
        print("=" * 55)

        self.vision = Vision(model=vision_model, debug=debug)
        self.vision_only = vision_only
        self.brain = None if vision_only else Brain(model=brain_model)
        self.actions = Actions()

        if game_region:
            self.actions.set_game_region(*game_region)

        self.game_rules = ""
        self.loop_count = 0
        self.stuck_count = 0

        print(f"⚡ {'Vision-only' if vision_only else 'Vision+Brain'} mode")

    def set_game_rules(self, rules):
        self.game_rules = rules

    def _handle_not_playing(self, game_state, base_img=None):
        """Find and click the right button when not playing."""
        state = game_state.get("state", "?")
        print(f"   📋 State: {state}")

        # Check if analysis already found buttons with cell numbers
        buttons = game_state.get("buttons", [])

        # If no buttons, do dedicated scan
        if not buttons:
            buttons, base_img = self.vision.find_buttons(
                region=self.actions.region,
                hint="Retry, Restart, Play, or Start button",
            )

        if not buttons:
            self.stuck_count += 1
            print(f"   ⚠️  No buttons found (stuck={self.stuck_count})")
            if self.stuck_count >= 3:
                # Try clicking screen center
                self.stuck_count = 0
                return {"action": "move_click", "click_x": 50, "click_y": 50,
                        "reason": "no buttons, trying center"}
            return {"action": "press_key", "key": "space",
                    "reason": "no buttons, trying space"}

        print(f"   🔘 Found {len(buttons)} button(s):")
        for b in buttons:
            print(f"      • \"{b.get('label','?')}\" in cell {b.get('cell','?')}")

        best = _best_button(buttons)
        if not best or "cell" not in best:
            return {"action": "press_key", "key": "space",
                    "reason": "no valid button cell"}

        cell = int(best["cell"])
        total = self.vision.grid_rows * self.vision.grid_cols
        if cell < 1 or cell > total:
            print(f"   ⚠️  Bad cell {cell}, using center")
            return {"action": "move_click", "click_x": 50, "click_y": 50,
                    "reason": "bad cell number"}

        # Refine the click within that cell
        if base_img is None:
            base_img = self.vision.capture(self.actions.region)

        x_pct, y_pct = self.vision._refine_click(
            base_img, cell, best.get("label", "button")
        )

        self.stuck_count = 0
        return {
            "action": "move_click",
            "click_x": x_pct,
            "click_y": y_pct,
            "reason": f"click '{best.get('label','?')}' (cell {cell})",
        }

    def step(self):
        self.loop_count += 1
        print(f"\n{'─'*20}  Loop {self.loop_count}  {'─'*20}")

        # ---- SEE ----
        t0 = time.time()
        game_state = self.vision.analyze(
            region=self.actions.region,
            game_context=self.game_rules,
        )
        print(f"👁️  {time.time()-t0:.1f}s")

        if not game_state:
            time.sleep(0.5)
            return None, {"action": "wait"}

        state = game_state.get("state", "unknown")

        # ---- DECIDE ----
        if state in ("menu", "game_over", "paused"):
            action = self._handle_not_playing(game_state)

        elif state == "playing":
            self.stuck_count = 0
            if self.vision_only:
                action = game_state  # use vision's action
            else:
                t0 = time.time()
                action = self.brain.decide(game_state, self.game_rules)
                print(f"🧠 {time.time()-t0:.1f}s")
        else:
            action = game_state

        # ---- ACT ----
        self.actions.execute(action)
        return game_state, action

    def run(self, max_loops=None, delay=0.5):
        print(f"\n🚀 Starting in 3…")
        for i in range(3, 0, -1):
            print(f"   {i}…")
            time.sleep(1)

        loop = 0
        try:
            while max_loops is None or loop < max_loops:
                self.step()
                time.sleep(delay)
                loop += 1
        except KeyboardInterrupt:
            print("\n⛔ Stopped")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe")
        print(f"📊 {loop} loops done")


# ============================================================
#  PRESETS
# ============================================================

PRESETS = {
    "flappy_bird": (
        "Flappy Bird: press_key space to flap. If bird low or pipe close, "
        "flap. If bird high, wait. On game_over find Retry button and click it."
    ),
    "cookie_clicker": "Click the big cookie. Buy cheapest upgrade.",
    "dino_run": "SPACE=jump over cacti. DOWN=duck under birds.",
    "2048": "Arrow keys. Keep highest tile in corner.",
}


# ============================================================
#  REGION PICKER
# ============================================================

def pick_region():
    """Click two corners to define game area."""
    print("\n🖱️  Move mouse to TOP-LEFT corner of game, press Enter")
    input("   → ")
    x1, y1 = pyautogui.position()
    print(f"   ({x1}, {y1})")

    print("   Move to BOTTOM-RIGHT corner, press Enter")
    input("   → ")
    x2, y2 = pyautogui.position()
    print(f"   ({x2}, {y2})")

    region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    print(f"   ✅ Region: {region}")
    return region


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n🎮 AI Game Player — Setup\n")

    presets = list(PRESETS.keys())
    for i, name in enumerate(presets, 1):
        print(f"  {i}. {name}")
    print(f"  {len(presets)+1}. custom")

    choice = input("\nGame: ").strip()
    if choice.isdigit() and int(choice) <= len(presets):
        rules = PRESETS[presets[int(choice) - 1]]
        print(f"✅ {presets[int(choice)-1]}")
    else:
        rules = input("Game rules: ").strip()

    v_model = input("Vision model [gemma3]: ").strip() or "gemma3"
    use_brain = input("Separate brain? (y/n) [n]: ").strip().lower() == "y"
    b_model = input("Brain model [mistral]: ").strip() or "mistral" if use_brain else "mistral"

    delay = float(input("Loop delay [0.3]: ").strip() or "0.3")
    max_l = input("Max loops [unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None

    reg = input("Region — (m)anual, (p)ick, (f)ullscreen [f]: ").strip().lower()
    region = None
    if reg == "m":
        try:
            region = tuple(int(p) for p in input("  x,y,w,h: ").split(","))
        except ValueError:
            pass
    elif reg == "p":
        region = pick_region()

    debug = input("Save debug screenshots? (y/n) [y]: ").strip().lower() != "n"

    ai = GameAI(
        vision_model=v_model, brain_model=b_model,
        game_region=region, vision_only=not use_brain, debug=debug,
    )
    ai.set_game_rules(rules)

    mode = input("(r)un or (s)tep? [r]: ").strip().lower()
    if mode == "s":
        while True:
            ai.step()
            if input("Next? (Enter/q): ").strip().lower() == "q":
                break
    else:
        ai.run(max_loops=max_loops, delay=delay)


if __name__ == "__main__":
    main()