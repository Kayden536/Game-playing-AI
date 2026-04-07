import ollama
import pyautogui
import cv2
import numpy as np
import easyocr
import io
import json
import time
import os
import re
from PIL import ImageGrab, Image, ImageEnhance

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

# ============================================================
#  INSTALL DEPS:
#    pip install easyocr opencv-python pyautogui pillow ollama
# ============================================================


# ============================================================
#  SCREEN READER — OCR + OpenCV
# ============================================================

class ScreenReader:
    BUTTON_KEYWORDS = [
        "play", "start", "retry", "restart", "again",
        "continue", "resume", "ok", "yes", "tap",
        "new game", "replay", "begin", "next", "menu",
        "try again", "play again", "submit", "insert coin",
        "revive", "collect", "buy", "upgrade", "hire",
    ]

    def __init__(self, gpu=True):
        print("📖 Loading OCR engine...")
        try:
            self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            print("📖 OCR ready")
        except Exception as e:
            print(f"📖 OCR GPU failed ({e}), trying CPU...")
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("📖 OCR ready (CPU)")

    def pil_to_cv(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def cv_to_pil(self, cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def find_all_text(self, pil_img):
        cv_img = self.pil_to_cv(pil_img)
        results = self.reader.readtext(cv_img)
        texts = []
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            w = x2 - x
            h = y2 - y
            texts.append({
                "text": text.strip(),
                "x": x, "y": y, "w": w, "h": h,
                "cx": x + w // 2,
                "cy": y + h // 2,
                "confidence": round(conf, 3),
            })
        return texts

    def find_buttons(self, pil_img, extra_keywords=None):
        all_text = self.find_all_text(pil_img)
        keywords = list(self.BUTTON_KEYWORDS)
        if extra_keywords:
            keywords.extend([k.lower() for k in extra_keywords])

        matches = []
        img_cx = pil_img.size[0] / 2
        img_cy = pil_img.size[1] / 2

        for t in all_text:
            text_lower = t["text"].lower().strip()
            if len(text_lower) < 1:
                continue
            score = 0
            matched_kw = ""
            for kw in keywords:
                if kw in text_lower or text_lower in kw:
                    s = 100 if text_lower == kw else 50
                    if s > score:
                        score = s
                        matched_kw = kw
            if score > 0:
                dist = ((t["cx"] - img_cx)**2 + (t["cy"] - img_cy)**2) ** 0.5
                max_dist = (img_cx**2 + img_cy**2) ** 0.5
                center_bonus = (1 - dist / max(max_dist, 1)) * 20
                score += center_bonus
                t["match_score"] = score
                t["matched_keyword"] = matched_kw
                matches.append(t)

        matches.sort(key=lambda m: (m["match_score"], m["confidence"]), reverse=True)
        return matches

    def find_rectangles(self, pil_img, min_area=1000, max_area=None):
        cv_img = self.pil_to_cv(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        if max_area is None:
            max_area = pil_img.size[0] * pil_img.size[1] * 0.5

        rects = []
        seen = set()

        for method in ["canny", "adaptive", "otsu"]:
            if method == "canny":
                edges = cv2.Canny(gray, 50, 150)
            elif method == "adaptive":
                edges = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            else:
                _, edges = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = w / max(h, 1)
                    if 1.2 < aspect < 8.0 and h > 15:
                        key = (x // 10, y // 10, w // 10, h // 10)
                        if key not in seen:
                            seen.add(key)
                            rects.append({
                                "x": x, "y": y, "w": w, "h": h,
                                "cx": x + w // 2,
                                "cy": y + h // 2,
                                "area": area,
                            })

        rects.sort(key=lambda r: r["area"], reverse=True)
        return rects

    def find_clickable(self, pil_img, keywords=None):
        buttons = self.find_buttons(pil_img, keywords)
        if buttons:
            best = buttons[0]
            print(f"   📖 OCR button: \"{best['text']}\" at "
                  f"({best['cx']},{best['cy']}) conf={best['confidence']}")
            return best["cx"], best["cy"]

        rects = self.find_rectangles(pil_img)
        if rects:
            img_cx = pil_img.size[0] // 2
            img_cy = pil_img.size[1] // 2
            rects.sort(key=lambda r: abs(r["cx"] - img_cx) + abs(r["cy"] - img_cy))
            best = rects[0]
            print(f"   📐 Rectangle at ({best['cx']},{best['cy']}) "
                  f"size={best['w']}x{best['h']}")
            return best["cx"], best["cy"]

        return None

    def detect_state_fast(self, pil_img, game_name=None):
        all_text = self.find_all_text(pil_img)
        all_lower = " ".join(t["text"].lower() for t in all_text)

        game_over_words = [
            "game over", "try again", "retry", "restart",
            "you died", "failed", "you lose", "defeated", "lost",
            "level failed", "mission failed",
        ]
        menu_words = [
            "tap to start", "press start", "main menu",
            "tap to play", "click to start", "press space to start",
        ]
        pause_words = ["pause", "paused", "resume", "continue"]

        best_state = "playing"
        best_confidence = 0

        for w in game_over_words:
            if w in all_lower:
                conf = max(
                    [t["confidence"] for t in all_text
                     if w in t["text"].lower()], default=0
                )
                if conf > best_confidence:
                    best_state = "game_over"
                    best_confidence = conf

        for w in menu_words:
            if w in all_lower:
                conf = max(
                    [t["confidence"] for t in all_text
                     if w in t["text"].lower()], default=0
                )
                if conf > best_confidence:
                    best_state = "menu"
                    best_confidence = conf

        for w in pause_words:
            if w in all_lower:
                conf = max(
                    [t["confidence"] for t in all_text
                     if w in t["text"].lower()], default=0
                )
                if conf > best_confidence:
                    best_state = "paused"
                    best_confidence = conf

        if best_state != "playing" and best_confidence < 0.5:
            return "playing", all_text

        return best_state, all_text


# ============================================================
#  GAME VISION — LLM decisions
# ============================================================

class GameVision:
    def __init__(self, model="gemma3"):
        self.model = model
        print(f"🧠 Vision model: {model}")

    def _img_to_bytes(self, pil_img, max_dim=512):
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def should_act(self, pil_img, game_context=""):
        img_bytes = self._img_to_bytes(pil_img)
        prompt = f"""You are playing a game. {game_context}

Look at the screenshot. Should the player press the action key RIGHT NOW?

Reply with ONLY this JSON:
{{"act": true, "reason": "3 words max"}}

If unsure say act=false. JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 60},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 act: {raw[:80]}")
            parsed = self._parse(raw)
            return parsed.get("act", False), parsed.get("reason", "")
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return False, "error"

    def choose_key(self, pil_img, game_context="", allowed_keys=None):
        if allowed_keys is None:
            allowed_keys = ["space", "up", "down", "left", "right"]
        img_bytes = self._img_to_bytes(pil_img)
        prompt = f"""You are playing a game. {game_context}

Available keys: {', '.join(allowed_keys)}

Which key should be pressed RIGHT NOW? Or wait?

Reply ONLY: {{"key": "space", "reason": "3 words"}}
Use "none" if you should wait. JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 60},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 key: {raw[:80]}")
            parsed = self._parse(raw)
            return parsed.get("key", "none"), parsed.get("reason", "")
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return "none", "error"

    def where_to_click(self, pil_img, game_context=""):
        img_bytes = self._img_to_bytes(pil_img)
        prompt = f"""You are playing a game. {game_context}

Where should the mouse click RIGHT NOW?

Reply ONLY:
{{"x": 0-100, "y": 0-100, "what": "brief label", "reason": "why", "right_click": false}}

x and y are percentages of screen width/height.
0,0 = top-left. 100,100 = bottom-right.
JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 100},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 click: {raw[:100]}")
            return self._parse(raw)
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return {}

    def strategy_move(self, pil_img, game_context="", allowed_keys=None):
        img_bytes = self._img_to_bytes(pil_img, max_dim=672)
        keys_hint = f"Available keys: {', '.join(allowed_keys)}" if allowed_keys else ""
        prompt = f"""You are playing a game. {game_context}
{keys_hint}

Study the game board carefully.
What is the single best move RIGHT NOW?

Reply ONLY:
{{
  "action": "click or right_click or key or type",
  "x": 0-100,
  "y": 0-100,
  "key": "key name if pressing a key",
  "text": "text to type if typing",
  "reason": "brief reason"
}}

JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 150},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 strategy: {raw[:150]}")
            return self._parse(raw)
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return {}

    def classify_state(self, pil_img, game_context=""):
        img_bytes = self._img_to_bytes(pil_img)
        prompt = f"""What state is this game in? {game_context}

Reply ONLY: {{"state": "playing or menu or game_over or paused"}}
JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 30},
            )
            raw = resp["message"]["content"].strip()
            parsed = self._parse(raw)
            return parsed.get("state", "unknown")
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return "unknown"

    def _parse(self, raw):
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
            s = raw.find("{")
            e = raw.rfind("}") + 1
            if s != -1 and e > s:
                try:
                    return json.loads(raw[s:e])
                except json.JSONDecodeError:
                    pass
        return {}


# ============================================================
#  SCREEN CAPTURE
# ============================================================

def capture_screen(region=None):
    if region:
        x, y, w, h = region
        bbox = (x, y, x + w, y + h)
    else:
        bbox = None
    return ImageGrab.grab(bbox=bbox)


# ============================================================
#  ACTIONS
# ============================================================

class Actions:
    def __init__(self, region=None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.region = region
        print(f"🎯 Screen: {self.screen_w}×{self.screen_h}")
        if region:
            print(f"🎯 Region: {region}")

    def img_to_screen(self, img_x, img_y, img_size):
        img_w, img_h = img_size
        if self.region:
            rx, ry, rw, rh = self.region
            sx = rx + int(img_x / img_w * rw)
            sy = ry + int(img_y / img_h * rh)
        else:
            sx = int(img_x / img_w * self.screen_w)
            sy = int(img_y / img_h * self.screen_h)
        sx = max(0, min(sx, self.screen_w - 1))
        sy = max(0, min(sy, self.screen_h - 1))
        return sx, sy

    def pct_to_screen(self, x_pct, y_pct):
        if self.region:
            rx, ry, rw, rh = self.region
            sx = rx + int((x_pct / 100) * rw)
            sy = ry + int((y_pct / 100) * rh)
        else:
            sx = int((x_pct / 100) * self.screen_w)
            sy = int((y_pct / 100) * self.screen_h)
        sx = max(0, min(sx, self.screen_w - 1))
        sy = max(0, min(sy, self.screen_h - 1))
        return sx, sy

    def click(self, sx, sy, duration=0.2, right=False):
        print(f"     🖱️  {'right-' if right else ''}click screen({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=duration)
        time.sleep(0.05)
        if right:
            pyautogui.rightClick(sx, sy)
        else:
            pyautogui.click(sx, sy)

    def click_img(self, img_x, img_y, img_size, duration=0.2, right=False):
        sx, sy = self.img_to_screen(img_x, img_y, img_size)
        self.click(sx, sy, duration=duration, right=right)
        return sx, sy

    def click_pct(self, x_pct, y_pct, duration=0.2, right=False):
        sx, sy = self.pct_to_screen(x_pct, y_pct)
        self.click(sx, sy, duration=duration, right=right)
        return sx, sy

    def press(self, key="space"):
        print(f"     ⌨️  '{key}'")
        pyautogui.press(key)

    def hotkey(self, *keys):
        print(f"     ⌨️  hotkey{keys}")
        pyautogui.hotkey(*keys)

    def typewrite(self, text, interval=0.08):
        print(f"     ⌨️  type '{text}'")
        pyautogui.typewrite(text, interval=interval)

    def double_click(self, sx, sy):
        print(f"     🖱️  double-click screen({sx},{sy})")
        pyautogui.doubleClick(sx, sy)

    def drag(self, x1, y1, x2, y2, duration=0.3):
        print(f"     🖱️  drag ({x1},{y1})→({x2},{y2})")
        pyautogui.moveTo(x1, y1, duration=0.1)
        pyautogui.dragTo(x2, y2, duration=duration, button="left")

    def scroll(self, clicks=3):
        print(f"     🖱️  scroll {clicks}")
        pyautogui.scroll(clicks)


# ============================================================
#  GAME PRESETS
# ============================================================

PRESETS = {

    # ── Single Key ──────────────────────────────────────────

    "flappy_bird": {
        "rules": (
            "Flappy Bird: the bird falls due to gravity. "
            "Click or press SPACE to flap upward. "
            "Fly through the gap between top and bottom pipes. "
            "If a pipe is approaching and the bird is at or below "
            "the gap center press space. If above the gap wait. "
            "React early — the game is fast."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["tap", "flap", "ok", "play"],
        "game_type": "single_key",
        "click_to_play": True,
        "action_cooldown": 2,
    },

    "geometry_dash": {
        "rules": (
            "Geometry Dash: the character moves right automatically. "
            "Press SPACE or click to jump over spikes and obstacles. "
            "Timing is critical — jump just before hitting an obstacle. "
            "Hold space to keep jumping on steep sections."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["play", "retry", "ok", "practice", "normal"],
        "game_type": "single_key",
        "click_to_play": True,
        "action_cooldown": 2,
    },

    "stickman_hook": {
        "rules": (
            "Stickman Hook: click/press to shoot a rope and attach "
            "to a circle peg. Swing and release at the right moment "
            "to fly toward the next peg. Click to grab, release to "
            "let go and fly."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["play", "retry", "next", "continue"],
        "game_type": "single_key",
        "click_to_play": True,
        "action_cooldown": 3,
    },

    # ── Multi Key ───────────────────────────────────────────

    "dino_run": {
        "rules": (
            "Chrome Dino Runner: press SPACE to jump over cacti on "
            "the ground. Press DOWN arrow to duck under flying "
            "pterodactyls. If a cactus is on the ground ahead press "
            "space. If a bird is at head height press down. "
            "Do nothing if the path is clear."
        ),
        "action_key": "space",
        "allowed_keys": ["space", "down"],
        "extra_buttons": [],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 2,
    },

    "subway_surfers": {
        "rules": (
            "Subway Surfers: avoid trains by changing lanes. "
            "Press LEFT or RIGHT to dodge sideways. "
            "Press UP to jump over barriers. "
            "Press DOWN to roll under obstacles. "
            "Collect coins on the way."
        ),
        "action_key": "up",
        "allowed_keys": ["up", "down", "left", "right"],
        "extra_buttons": ["play", "retry", "revive", "continue"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 2,
    },

    "2048": {
        "rules": (
            "2048: slide all tiles using arrow keys. "
            "When two tiles with the same number collide they merge. "
            "Strategy: keep the highest tile in the bottom-left corner. "
            "Build a descending chain along the bottom row. "
            "Prefer LEFT then DOWN moves. "
            "Avoid UP unless necessary."
        ),
        "action_key": "left",
        "allowed_keys": ["up", "down", "left", "right"],
        "extra_buttons": ["new game", "new", "try again", "keep going"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 4,
    },

    "snake": {
        "rules": (
            "Snake: control the snake with arrow keys. "
            "Eat the bright food dot to grow. "
            "Do NOT hit the walls or your own body. "
            "Look at where the food is relative to the snake head. "
            "Move toward food while avoiding your own tail. "
            "Plan ahead so you don't trap yourself."
        ),
        "action_key": "right",
        "allowed_keys": ["up", "down", "left", "right"],
        "extra_buttons": ["play", "restart", "new game", "start"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 3,
    },

    "tetris": {
        "rules": (
            "Tetris: falling blocks must be arranged to complete "
            "horizontal lines which then disappear. "
            "LEFT/RIGHT arrow to move the piece sideways. "
            "UP arrow to rotate the piece. "
            "DOWN arrow to soft drop faster. "
            "SPACE to hard drop instantly. "
            "Fill gaps at the bottom. Keep the stack as low as possible."
        ),
        "action_key": "down",
        "allowed_keys": ["left", "right", "up", "down", "space"],
        "extra_buttons": ["start", "play", "new game", "ok"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 2,
    },

    "pacman": {
        "rules": (
            "Pac-Man: eat all dots to complete the level. "
            "Use arrow keys to navigate the maze. "
            "Avoid ghosts — being touched kills you. "
            "Eat large power pellets to turn ghosts blue "
            "then you can eat them for bonus points. "
            "Plan a path to collect dots while avoiding ghosts."
        ),
        "action_key": "right",
        "allowed_keys": ["up", "down", "left", "right"],
        "extra_buttons": ["start", "play", "insert coin", "1 player"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 2,
    },

    "space_invaders": {
        "rules": (
            "Space Invaders: shoot the descending aliens. "
            "Press LEFT/RIGHT to move your ship. "
            "Press SPACE to shoot. "
            "Hide behind barriers to avoid alien bullets. "
            "Move away from aliens directly above you. "
            "Shoot the nearest or lowest alien."
        ),
        "action_key": "space",
        "allowed_keys": ["left", "right", "space"],
        "extra_buttons": ["play", "start", "insert coin", "1 player"],
        "game_type": "multi_key",
        "click_to_play": False,
        "action_cooldown": 2,
    },

    # ── Clicker ─────────────────────────────────────────────

    "cookie_clicker": {
        "rules": (
            "Cookie Clicker: click the large round cookie on the "
            "LEFT side of the screen to earn cookies. "
            "The cookie is the biggest circular object on the left half. "
            "If a golden cookie appears anywhere on screen click it "
            "immediately for a big bonus. "
            "Occasionally check the SHOP on the right side — "
            "buy the cheapest available upgrade or building."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["buy", "upgrade", "accept", "fortune"],
        "game_type": "clicker",
        "click_to_play": True,
        "action_cooldown": 1,
        "click_target": "cookie",
    },

    "idle_miner": {
        "rules": (
            "Idle Miner Tycoon: tap to collect cash bags when they "
            "appear. Buy upgrades and managers to automate mining. "
            "Tap any collect buttons or boost timers. "
            "Hire managers so shafts run automatically."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["collect", "boost", "upgrade", "hire",
                          "buy", "claim", "free", "watch"],
        "game_type": "clicker",
        "click_to_play": True,
        "action_cooldown": 2,
    },

    # ── Strategy ────────────────────────────────────────────

    "minesweeper": {
        "rules": (
            "Minesweeper: left-click to reveal a safe cell. "
            "Right-click to flag a cell you think has a mine. "
            "Numbers show how many mines are in the 8 adjacent cells. "
            "Start by clicking the center of the board. "
            "Use number logic to determine safe cells. "
            "If a number's mine count is already flagged, "
            "the remaining neighbours are safe to click."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["new game", "beginner", "intermediate",
                          "expert", "ok", "reset"],
        "game_type": "strategy",
        "click_to_play": True,
        "action_cooldown": 5,
    },

    "chess": {
        "rules": (
            "Chess: click a piece to select it then click the "
            "destination square. "
            "Control the center with pawns and knights early. "
            "Develop bishops and knights before the queen. "
            "Castle early to protect your king. "
            "Look for checkmate opportunities and avoid hanging pieces."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["new game", "play", "resign", "ok",
                          "accept", "white", "black"],
        "game_type": "strategy",
        "click_to_play": True,
        "action_cooldown": 8,
    },

    "solitaire": {
        "rules": (
            "Klondike Solitaire: build foundation piles A→K per suit. "
            "Stack tableau cards in descending order, alternating colors. "
            "Click the deck (top-left) to draw new cards. "
            "Move kings to empty columns. "
            "Always move cards to foundation if possible. "
            "Reveal hidden cards by moving stacks."
        ),
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["new game", "deal", "undo", "new",
                          "restart", "ok"],
        "game_type": "strategy",
        "click_to_play": True,
        "action_cooldown": 6,
    },

    "wordle": {
        "rules": (
            "Wordle: guess a 5-letter word in 6 tries. "
            "Green letter = correct letter correct position. "
            "Yellow letter = correct letter wrong position. "
            "Gray letter = letter not in the word. "
            "Start with a strong word like CRANE or AUDIO. "
            "Type your 5-letter guess then press ENTER."
        ),
        "action_key": "enter",
        "allowed_keys": None,
        "extra_buttons": ["play", "new game", "statistics", "ok"],
        "game_type": "typing",
        "click_to_play": False,
        "action_cooldown": 10,
    },
}


# ============================================================
#  GAME AI — main controller
# ============================================================

class GameAI:
    def __init__(self, vision_model="gemma3", game_region=None,
                 action_key="space", allowed_keys=None,
                 game_type="single_key", debug=True,
                 game_name="custom", action_cooldown=3,
                 extra_buttons=None, click_to_play=False,
                 preset_data=None):

        print("=" * 55)
        print("  🎮  AI Game Player — OCR + LLM")
        print("=" * 55)

        self.screen_reader = ScreenReader(gpu=True)
        self.game_vision = GameVision(model=vision_model)
        self.actions = Actions(region=game_region)
        self.region = game_region

        self.action_key = action_key
        self.allowed_keys = allowed_keys
        self.game_type = game_type
        self.game_rules = ""
        self.game_name = game_name
        self.debug = debug
        self.action_cooldown = action_cooldown
        self.extra_button_keywords = extra_buttons or []
        self.click_to_play = click_to_play
        self.preset_data = preset_data or {}

        self.loop_count = 0
        self.stuck_count = 0
        self.last_action_frame = 0
        self.last_state = "unknown"
        self.consecutive_same_state = 0

        if debug:
            os.makedirs("debug_frames", exist_ok=True)

        print(f"\n   Game      : {game_name}")
        print(f"   Type      : {game_type}")
        print(f"   Action key: {action_key}")
        print(f"   Multi-key : {allowed_keys}")
        print(f"   Cooldown  : {action_cooldown} loops")
        print(f"   Debug     : {'ON' if debug else 'OFF'}\n")

    def set_game_rules(self, rules):
        self.game_rules = rules

    def add_button_keywords(self, keywords):
        self.extra_button_keywords.extend([k.lower() for k in keywords])

    # ----------------------------------------------------------
    #  Debug helpers
    # ----------------------------------------------------------

    def _save_debug(self, pil_img, tag):
        if self.debug:
            path = f"debug_frames/{self.loop_count:04d}_{tag}.png"
            pil_img.save(path)

    def _save_debug_annotated(self, pil_img, texts, tag):
        if not self.debug:
            return
        cv_img = self.screen_reader.pil_to_cv(pil_img)
        for t in texts:
            x, y, w, h = t["x"], t["y"], t["w"], t["h"]
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{t['text']}({t['confidence']:.2f})",
                        (x, max(y - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        out = self.screen_reader.cv_to_pil(cv_img)
        out.save(f"debug_frames/{self.loop_count:04d}_{tag}.png")

    # ----------------------------------------------------------
    #  Button handler (menus / game over)
    # ----------------------------------------------------------

    def _handle_buttons(self, screenshot):
        buttons = self.screen_reader.find_buttons(
            screenshot, self.extra_button_keywords
        )
        if buttons:
            best = buttons[0]
            print(f"   🔘 Button: \"{best['text']}\" "
                  f"score={best['match_score']:.0f} "
                  f"conf={best['confidence']:.2f}")
            self.actions.click_img(best["cx"], best["cy"], screenshot.size)
            self.stuck_count = 0
            time.sleep(0.5)
            return True

        target = self.screen_reader.find_clickable(
            screenshot, self.extra_button_keywords
        )
        if target:
            cx, cy = target
            self.actions.click_img(cx, cy, screenshot.size)
            self.stuck_count = 0
            time.sleep(0.5)
            return True

        self.stuck_count += 1
        print(f"   ⚠️  No button found (stuck={self.stuck_count})")

        if self.stuck_count >= 3:
            print("   ⚠️  Trying space + click center as fallback")
            self.actions.press("space")
            time.sleep(0.3)
            if self.region:
                _, _, rw, rh = self.region
                cx_pct, cy_pct = 50, 50
            else:
                cx_pct, cy_pct = 50, 50
            self.actions.click_pct(cx_pct, cy_pct)
            self.stuck_count = 0

        return False

    # ----------------------------------------------------------
    #  Single-key gameplay
    # ----------------------------------------------------------

    def _handle_single_key(self, screenshot):
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            print(f"   ⏳ Key cooldown")
            return

        should, reason = self.game_vision.should_act(
            screenshot, self.game_rules
        )
        if should:
            print(f"   🎮 PRESS '{self.action_key}' — {reason}")
            if self.click_to_play:
                # Also click for games that need click not keypress
                self.actions.click_pct(50, 50)
            else:
                self.actions.press(self.action_key)
            self.last_action_frame = self.loop_count
        else:
            print(f"   🎮 Wait — {reason}")

    # ----------------------------------------------------------
    #  Multi-key gameplay
    # ----------------------------------------------------------

    def _handle_multi_key(self, screenshot):
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            print(f"   ⏳ Key cooldown")
            return

        key, reason = self.game_vision.choose_key(
            screenshot, self.game_rules, self.allowed_keys
        )
        if key and key.lower() not in ("none", "wait", ""):
            print(f"   🎮 PRESS '{key}' — {reason}")
            self.actions.press(key)
            self.last_action_frame = self.loop_count
        else:
            print(f"   🎮 Wait — {reason}")

    # ----------------------------------------------------------
    #  Clicker gameplay
    # ----------------------------------------------------------

    def _handle_clicker(self, screenshot):
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            print(f"   ⏳ Click cooldown")
            return

        # Special handling for cookie clicker — find the cookie
        if self.game_name == "cookie_clicker":
            self._handle_cookie_clicker(screenshot)
            return

        # General clicker — ask LLM where to click
        parsed = self.game_vision.where_to_click(screenshot, self.game_rules)
        if parsed:
            x_pct = parsed.get("x", 50)
            y_pct = parsed.get("y", 50)
            what = parsed.get("what", "?")
            reason = parsed.get("reason", "")
            right = parsed.get("right_click", False)
            print(f"   🖱️  Click '{what}' at {x_pct}%,{y_pct}% — {reason}")
            self.actions.click_pct(x_pct, y_pct, right=right)
            self.last_action_frame = self.loop_count
        else:
            # Fallback: click center
            print("   🖱️  Fallback click center")
            self.actions.click_pct(50, 50)
            self.last_action_frame = self.loop_count

    def _handle_cookie_clicker(self, screenshot):
        """Dedicated Cookie Clicker handler.
        Clicks the cookie rapidly; also watches for golden cookies."""

        img_w, img_h = screenshot.size
        cv_img = self.screen_reader.pil_to_cv(screenshot)

        # --- Find golden cookie first (priority) ---
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        gold_lower = np.array([15, 100, 100])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
        gold_mask[:, :img_w // 2] = 0  # golden cookie NOT on left half
        contours, _ = cv2.findContours(
            gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    gx = int(M["m10"] / M["m00"])
                    gy = int(M["m01"] / M["m00"])
                    print(f"   🌟 Golden cookie at ({gx},{gy})!")
                    self.actions.click_img(gx, gy, screenshot.size)
                    self.last_action_frame = self.loop_count
                    return

        # --- Click the main cookie (left ~25% of screen) ---
        # The cookie is a large brown/tan circular object on the left
        left_half = screenshot.crop((0, 0, img_w // 2, img_h))
        left_cv = self.screen_reader.pil_to_cv(left_half)

        # Look for large circular blob on the left
        gray = cv2.cvtColor(left_cv, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=img_h // 2,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Pick largest circle
            best = sorted(circles[0], key=lambda c: c[2], reverse=True)[0]
            cx, cy, r = int(best[0]), int(best[1]), int(best[2])
            print(f"   🍪 Cookie circle at left({cx},{cy}) r={r}")
            self.actions.click_img(cx, cy, left_half.size)
            self.last_action_frame = self.loop_count
            return

        # Fallback: click left-center
        print("   🍪 Cookie fallback: click left-center")
        self.actions.click_pct(20, 50)
        self.last_action_frame = self.loop_count

    # ----------------------------------------------------------
    #  Strategy gameplay
    # ----------------------------------------------------------

    def _handle_strategy(self, screenshot):
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            print(f"   ⏳ Strategy cooldown")
            return

        parsed = self.game_vision.strategy_move(
            screenshot, self.game_rules, self.allowed_keys
        )

        if not parsed:
            print("   ⚠️  No strategy move returned")
            return

        action = parsed.get("action", "click")
        x_pct = parsed.get("x", 50)
        y_pct = parsed.get("y", 50)
        key = parsed.get("key", "")
        text = parsed.get("text", "")
        reason = parsed.get("reason", "")

        print(f"   🧩 Strategy: {action} — {reason}")

        if action in ("click", "left_click"):
            self.actions.click_pct(x_pct, y_pct)

        elif action == "right_click":
            self.actions.click_pct(x_pct, y_pct, right=True)

        elif action == "key":
            if key:
                self.actions.press(key)

        elif action == "type":
            if text:
                # Click first to make sure the game has focus
                self.actions.click_pct(x_pct, y_pct)
                time.sleep(0.2)
                self.actions.typewrite(text.lower())

        elif action == "double_click":
            sx, sy = self.actions.pct_to_screen(x_pct, y_pct)
            self.actions.double_click(sx, sy)

        else:
            # Default to click
            self.actions.click_pct(x_pct, y_pct)

        self.last_action_frame = self.loop_count

    # ----------------------------------------------------------
    #  Typing game (Wordle etc.)
    # ----------------------------------------------------------

    def _handle_typing_game(self, screenshot):
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            print(f"   ⏳ Typing cooldown")
            return

        # For Wordle: ask LLM for the best word guess
        img_bytes = self.game_vision._img_to_bytes(screenshot, max_dim=512)
        prompt = f"""You are playing Wordle. {self.game_rules}

Look at the board carefully.
- Green letters are correct position
- Yellow letters are in the word but wrong position  
- Gray letters are not in the word

What 5-letter word should you guess next?
If this is the start of the game, suggest CRANE or AUDIO.

Reply ONLY: {{"word": "CRANE", "reason": "why"}}
JSON only. Word must be exactly 5 letters, all caps."""

        try:
            resp = ollama.chat(
                model=self.game_vision.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.2, "num_predict": 80},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 Wordle: {raw[:100]}")
            parsed = self.game_vision._parse(raw)
            word = parsed.get("word", "CRANE").upper().strip()
            reason = parsed.get("reason", "")

            # Validate: must be 5 letters
            word = re.sub(r"[^A-Z]", "", word)[:5]
            if len(word) != 5:
                word = "CRANE"

            print(f"   🔤 Typing '{word}' — {reason}")
            # Click the game area first for focus
            self.actions.click_pct(50, 50)
            time.sleep(0.3)
            self.actions.typewrite(word.lower())
            time.sleep(0.2)
            self.actions.press("enter")
            self.last_action_frame = self.loop_count

        except Exception as e:
            print(f"   ❌ Typing error: {e}")

    # ----------------------------------------------------------
    #  Main step
    # ----------------------------------------------------------

    def step(self):
        self.loop_count += 1
        print(f"\n{'─'*22}  Loop {self.loop_count}  {'─'*22}")

        # 1. CAPTURE
        screenshot = capture_screen(self.region)
        self._save_debug(screenshot, "raw")

        # 2. FAST STATE DETECTION via OCR
        t0 = time.time()
        state, all_text = self.screen_reader.detect_state_fast(
            screenshot, game_name=self.game_name
        )
        ocr_ms = (time.time() - t0) * 1000
        print(f"   📖 OCR {ocr_ms:.0f}ms | state={state} | "
              f"{len(all_text)} regions")

        if all_text:
            preview = [f'"{t["text"]}"' for t in all_text[:6]]
            print(f"   📖 Text: {', '.join(preview)}")

        self._save_debug_annotated(screenshot, all_text, "ocr")

        # 3. TRACK STUCK
        if state == self.last_state:
            self.consecutive_same_state += 1
        else:
            self.consecutive_same_state = 0
        self.last_state = state

        # 4. HANDLE STATE
        if state in ("game_over", "menu", "paused"):
            print(f"   📋 {state.upper()} → finding buttons...")
            time.sleep(0.5)
            self._handle_buttons(screenshot)

        elif state == "playing":
            gt = self.game_type
            print(f"   📋 PLAYING | type={gt}")

            if gt == "single_key":
                self._handle_single_key(screenshot)
            elif gt == "multi_key":
                self._handle_multi_key(screenshot)
            elif gt == "clicker":
                self._handle_clicker(screenshot)
            elif gt == "strategy":
                self._handle_strategy(screenshot)
            elif gt == "typing":
                self._handle_typing_game(screenshot)
            else:
                self._handle_single_key(screenshot)

        else:
            # Uncertain — ask LLM
            print(f"   📋 Uncertain — asking LLM...")
            llm_state = self.game_vision.classify_state(
                screenshot, self.game_rules
            )
            print(f"   🧠 LLM state: {llm_state}")

            if llm_state in ("game_over", "menu", "paused"):
                self._handle_buttons(screenshot)
            elif llm_state == "playing":
                gt = self.game_type
                if gt == "single_key":
                    self._handle_single_key(screenshot)
                elif gt == "multi_key":
                    self._handle_multi_key(screenshot)
                elif gt == "clicker":
                    self._handle_clicker(screenshot)
                elif gt == "strategy":
                    self._handle_strategy(screenshot)
                elif gt == "typing":
                    self._handle_typing_game(screenshot)
            else:
                # Final fallback
                print("   ⚠️  Unknown state — pressing action key")
                self.actions.press(self.action_key)

    # ----------------------------------------------------------
    #  Run loop
    # ----------------------------------------------------------

    def run(self, max_loops=None, delay=0.3):
        print(f"\n🚀 Starting in 3…")
        for i in range(3, 0, -1):
            print(f"   {i}…")
            time.sleep(1)
        print("   GO!\n")

        loop = 0
        try:
            while max_loops is None or loop < max_loops:
                self.step()
                time.sleep(delay)
                loop += 1
        except KeyboardInterrupt:
            print("\n⛔ Stopped by user")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe triggered (mouse top-left)")

        print(f"\n📊 {loop} loops completed")


# ============================================================
#  REGION PICKER
# ============================================================

def pick_region():
    print("\n🖱️  Move mouse to TOP-LEFT of game window then press Enter")
    input("   → ")
    x1, y1 = pyautogui.position()
    print(f"   ({x1}, {y1})")
    print("   Move mouse to BOTTOM-RIGHT of game window then press Enter")
    input("   → ")
    x2, y2 = pyautogui.position()
    print(f"   ({x2}, {y2})")
    region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    print(f"   ✅ Region: {region}")
    return region


# ============================================================
#  OCR TEST
# ============================================================

def test_ocr(region=None):
    print("\n🧪 OCR Test\n")
    reader = ScreenReader(gpu=True)
    screenshot = capture_screen(region)
    print(f"   Size: {screenshot.size}")

    texts = reader.find_all_text(screenshot)
    print(f"\n   Text ({len(texts)}):")
    for t in texts:
        print(f"      \"{t['text']}\" @ ({t['cx']},{t['cy']}) "
              f"conf={t['confidence']:.2f}")

    buttons = reader.find_buttons(screenshot)
    print(f"\n   Buttons ({len(buttons)}):")
    for b in buttons:
        print(f"      \"{b['text']}\" @ ({b['cx']},{b['cy']}) "
              f"kw='{b['matched_keyword']}' score={b['match_score']:.0f}")

    rects = reader.find_rectangles(screenshot)
    print(f"\n   Rectangles (top 5):")
    for r in rects[:5]:
        print(f"      ({r['cx']},{r['cy']}) {r['w']}×{r['h']}")

    state, _ = reader.detect_state_fast(screenshot)
    print(f"\n   State: {state}")

    # Annotate and save
    cv_img = reader.pil_to_cv(screenshot)
    for t in texts:
        cv2.rectangle(cv_img, (t["x"], t["y"]),
                      (t["x"] + t["w"], t["y"] + t["h"]), (0, 255, 0), 2)
        cv2.putText(cv_img, t["text"], (t["x"], max(t["y"] - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for b in buttons:
        cv2.circle(cv_img, (b["cx"], b["cy"]), 12, (0, 0, 255), -1)
    out = reader.cv_to_pil(cv_img)
    out.save("debug_ocr_test.png")
    print("\n   Saved: debug_ocr_test.png")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n🎮 AI Game Player\n")
    print("Modes:")
    print("  1. Play a game")
    print("  2. Test OCR on screen")
    mode = input("\nChoice [1]: ").strip() or "1"

    if mode == "2":
        reg = input("Region? (p)ick / (f)ullscreen [f]: ").strip().lower()
        region = pick_region() if reg == "p" else None
        test_ocr(region)
        return

    # ── Game selection ──
    presets = list(PRESETS.keys())
    print("\nAvailable games:")
    for i, name in enumerate(presets, 1):
        gtype = PRESETS[name].get("game_type", "?")
        print(f"  {i:2d}. {name:<20} [{gtype}]")
    print(f"  {len(presets)+1:2d}. custom")

    choice = input("\nGame: ").strip()
    preset = None
    selected_name = "custom"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            selected_name = presets[idx]
            preset = PRESETS[selected_name]
            print(f"✅ {selected_name}")

    if preset:
        rules        = preset["rules"]
        action_key   = preset["action_key"]
        allowed_keys = preset["allowed_keys"]
        extra_buttons= preset.get("extra_buttons", [])
        game_type    = preset.get("game_type", "single_key")
        cooldown     = preset.get("action_cooldown", 3)
        click_play   = preset.get("click_to_play", False)
    else:
        rules        = input("Game rules: ").strip()
        action_key   = input("Action key [space]: ").strip() or "space"
        mk           = input("Multiple keys? comma-sep or empty: ").strip()
        allowed_keys = [k.strip() for k in mk.split(",")] if mk else None
        gt_input     = input("Game type (single_key/multi_key/clicker/strategy/typing) [single_key]: ").strip()
        game_type    = gt_input if gt_input else "single_key"
        cooldown     = int(input("Action cooldown loops [3]: ").strip() or "3")
        click_play   = input("Click to play? (y/n) [n]: ").strip().lower() == "y"
        eb           = input("Extra button keywords comma-sep: ").strip()
        extra_buttons= [k.strip() for k in eb.split(",")] if eb else []

    # ── Vision model ──
    v_model = input("Vision model [gemma3]: ").strip() or "gemma3"

    # ── Timing ──
    delay  = float(input("Loop delay seconds [0.3]: ").strip() or "0.3")
    max_l  = input("Max loops [unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None

    # ── Region ──
    reg = input("Region? (m)anual / (p)ick / (f)ullscreen [f]: ").strip().lower()
    region = None
    if reg == "m":
        try:
            region = tuple(int(p) for p in input("  x,y,w,h: ").split(","))
        except ValueError:
            print("  Bad format, using fullscreen")
    elif reg == "p":
        region = pick_region()

    debug = input("Debug frames? (y/n) [y]: ").strip().lower() != "n"

    # ── Build AI ──
    ai = GameAI(
        vision_model=v_model,
        game_region=region,
        action_key=action_key,
        allowed_keys=allowed_keys,
        game_type=game_type,
        debug=debug,
        game_name=selected_name,
        action_cooldown=cooldown,
        extra_buttons=extra_buttons,
        click_to_play=click_play,
        preset_data=preset or {},
    )
    ai.set_game_rules(rules)

    # ── Run ──
    run_mode = input("(r)un / (s)tep [r]: ").strip().lower()
    if run_mode == "s":
        while True:
            ai.step()
            if input("\nNext? (Enter/q): ").strip().lower() == "q":
                break
    else:
        ai.run(max_loops=max_loops, delay=delay)


if __name__ == "__main__":
    main()