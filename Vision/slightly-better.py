import ollama
import pyautogui
import cv2
import numpy as np
import easyocr
import io
import json
import time
import os
from PIL import ImageGrab, Image, ImageEnhance

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05
 
# ============================================================
#  INSTALL DEPS:
#    pip install easyocr opencv-python pyautogui pillow ollama
#
#  easyocr downloads a model (~100MB) on first run.
#  It uses torch but you already have GPU for ollama.
# ============================================================


# ============================================================
#  SCREEN READER — OCR + OpenCV (fast, pixel-perfect)
# ============================================================

class ScreenReader:
    """Finds text, buttons, and UI elements using OCR and
    computer vision. No LLM needed. Fast and accurate."""

    # Common button labels across many games
    BUTTON_KEYWORDS = [
        "play", "start", "retry", "restart", "again",
        "continue", "resume", "ok", "yes", "tap",
        "new game", "replay", "begin", "next", "menu",
        "try again", "play again","submit",
    ]

    def __init__(self, gpu=True):
        print("📖 Loading OCR engine...")
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        print("📖 OCR ready")

    def pil_to_cv(self, pil_img):
        """PIL Image → OpenCV BGR numpy array."""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def cv_to_pil(self, cv_img):
        """OpenCV BGR → PIL Image."""
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------------------
    #  OCR — find all text with bounding boxes
    # ----------------------------------------------------------
    def find_all_text(self, pil_img):
        """Returns list of dicts:
        [{"text": "RETRY", "x": 340, "y": 210, "w": 120, "h": 40,
          "cx": 400, "cy": 230, "confidence": 0.95}, ...]

        Coordinates are in the image's pixel space."""
        cv_img = self.pil_to_cv(pil_img)
        results = self.reader.readtext(cv_img)

        texts = []
        for bbox, text, conf in results:
            # bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
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
                "cx": x + w // 2,       # center x
                "cy": y + h // 2,       # center y
                "confidence": round(conf, 3),
            })

        return texts

    # ----------------------------------------------------------
    #  Find buttons by matching text against keywords
    # ----------------------------------------------------------
    def find_buttons(self, pil_img, extra_keywords=None):
        """Find text regions that look like clickable buttons.
        Returns list sorted by relevance (best match first)."""
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

            # Check for keyword match
            score = 0
            matched_kw = ""
            for kw in keywords:
                if kw in text_lower or text_lower in kw:
                    # Exact match scores higher
                    s = 100 if text_lower == kw else 50
                    if s > score:
                        score = s
                        matched_kw = kw

            if score > 0:
                dist = ((t["cx"] - img_cx)**2 + (t["cy"] - img_cy)**2) ** 0.5
                max_dist = (img_cx**2 + img_cy**2) ** 0.5
                center_bonus = (1 - dist/max_dist) * 20
                score += center_bonus
                
                t["match_score"] = score
                t["matched_keyword"] = matched_kw
                matches.append(t)

        # Sort: highest score first, then highest confidence
        matches.sort(key=lambda m: (m["match_score"], m["confidence"]),
                     reverse=True)
        return matches

    # ----------------------------------------------------------
    #  Find rectangular button shapes (when OCR misses text)
    # ----------------------------------------------------------
    def find_rectangles(self, pil_img, min_area=1000, max_area=None):
        """Find rectangular contours that could be buttons.
        Returns list of {"x","y","w","h","cx","cy","area"}."""
        cv_img = self.pil_to_cv(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        if max_area is None:
            max_area = pil_img.size[0] * pil_img.size[1] * 0.5

        # Try multiple edge detection approaches
        rects = []
        seen = set()

        for method in ["canny", "adaptive", "otsu"]:
            if method == "canny":
                edges = cv2.Canny(gray, 50, 150)
            elif method == "adaptive":
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                edges = thresh
            else:
                _, edges = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

            # Dilate to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                # Approximate polygon
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                # Rectangles have 4 vertices
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = w / max(h, 1)

                    # Button-like aspect ratio (not too thin, not too square)
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

    # ----------------------------------------------------------
    #  Combined: find best clickable target
    # ----------------------------------------------------------
    def find_clickable(self, pil_img, keywords=None):
        """Best effort: try OCR buttons first, then rectangles.
        Returns (cx, cy) in image pixel coords, or None."""

        # 1. Try OCR
        buttons = self.find_buttons(pil_img, keywords)
        if buttons:
            best = buttons[0]
            print(f"   📖 OCR found: \"{best['text']}\" at "
                  f"({best['cx']}, {best['cy']}) "
                  f"conf={best['confidence']}")
            return best["cx"], best["cy"]

        # 2. Try rectangle detection
        rects = self.find_rectangles(pil_img)
        if rects:
            # Pick the rectangle closest to center of image
            img_cx = pil_img.size[0] // 2
            img_cy = pil_img.size[1] // 2
            rects.sort(key=lambda r: abs(r["cx"] - img_cx) + abs(r["cy"] - img_cy))
            best = rects[0]
            print(f"   📐 Rectangle found at ({best['cx']}, {best['cy']}) "
                  f"size={best['w']}x{best['h']}")
            return best["cx"], best["cy"]

        return None

    # ----------------------------------------------------------
    #  Quick game state heuristics (no LLM needed)
    # ----------------------------------------------------------
    def detect_state_fast(self, pil_img, game_name=None):
        """Use OCR to quickly guess game state without LLM."""
        all_text = self.find_all_text(pil_img)
        all_lower = " ".join(t["text"].lower() for t in all_text)

        # Do NOT use generic words like score/best for game over
        game_over_words = [
            "game over", "try again", "retry", "restart",
            "you died", "failed", "you lose", "defeated", "lost"
        ]
        menu_words = [
            "play", "start", "begin", "new game", "tap to start",
            "press start", "main menu"
        ]
        pause_words = ["pause", "paused", "resume", "continue"]

        best_state = "playing"
        best_confidence = 0
        
        for w in game_over_words:
            if w in all_lower:
                # Check if high-confidence OCR
                conf = max([t["confidence"] for t in all_text 
                           if w in t["text"].lower()], default=0)
                if conf > best_confidence:
                    best_state = "game_over"
                    best_confidence = conf

        for w in menu_words:
            if w in all_lower:
                conf = max([t["confidence"] for t in all_text 
                           if w in t["text"].lower()], default=0)
                if conf > best_confidence:
                    best_state = "menu"
                    best_confidence = conf

        for w in pause_words:
            if w in all_lower:
                conf = max([t["confidence"] for t in all_text 
                           if w in t["text"].lower()], default=0)
                if conf > best_confidence:
                    best_state = "paused"
                    best_confidence = conf

        # Only return non-playing states if confidence > threshold
        if best_state != "playing" and best_confidence < 0.5:
            # Low confidence - might be false positive
            return "playing", all_text
        
        return best_state, all_text


# ============================================================
#  GAME VISION — LLM for gameplay decisions only
# ============================================================

class GameVision:
    """Uses VLM for high-level gameplay decisions.
    NOT used for finding buttons or coordinates."""

    def __init__(self, model="gemma3"):
        self.model = model
        print(f"🧠 Game vision: {model}")

    def _img_to_bytes(self, pil_img, max_dim=512):
        """Resize and convert to PNG bytes."""
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
        """Simple YES/NO — should the player press the action key?
        Much faster and more reliable than asking for coordinates."""
        img_bytes = self._img_to_bytes(pil_img)

        prompt = f"""You are playing a game. {game_context}

Look at the screenshot. Should the player press the action key RIGHT NOW?

Consider:
- Is there an obstacle approaching?
- Is the player in danger?
- Does the player need to act immediately?

Reply with ONLY this JSON:
{{"act": true, "reason": "3 words max"}}

If unsure, say act=false. JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 50},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 LLM: {raw[:80]}")
            parsed = self._parse(raw)
            return parsed.get("act", False), parsed.get("reason", "")
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return False, "error"

    def classify_state(self, pil_img, game_context=""):
        """Ask LLM what state the game is in. Only used as fallback
        when OCR-based detection is uncertain."""
        img_bytes = self._img_to_bytes(pil_img)

        prompt = f"""What state is this game in? {game_context}

Reply with ONLY this JSON:
{{"state": "playing" or "menu" or "game_over" or "paused"}}

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

    def choose_key(self, pil_img, game_context="", allowed_keys=None):
        """For games with multiple keys (arrows, etc.), ask which to press."""
        if allowed_keys is None:
            allowed_keys = ["space", "up", "down", "left", "right"]

        img_bytes = self._img_to_bytes(pil_img)

        prompt = f"""You are playing a game. {game_context}

Available keys: {', '.join(allowed_keys)}

Which key should be pressed RIGHT NOW? Or should we wait?

Reply with ONLY: {{"key": "space", "reason": "3 words"}}
Use "none" for the key if you should wait.
JSON only."""

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_bytes],
                }],
                options={"temperature": 0.1, "num_predict": 50},
            )
            raw = resp["message"]["content"].strip()
            print(f"   🧠 Key: {raw[:80]}")
            parsed = self._parse(raw)
            return parsed.get("key", "none"), parsed.get("reason", "")
        except Exception as e:
            print(f"   🧠 Error: {e}")
            return "none", "error"

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
#  SCREEN CAPTURE helper
# ============================================================

def capture_screen(region=None):
    """Grab screen. region = (x, y, w, h) or None.
    Returns PIL Image."""
    if region:
        x, y, w, h = region
        bbox = (x, y, x + w, y + h)  # ImageGrab wants left,top,right,bottom
    else:
        bbox = None
    return ImageGrab.grab(bbox=bbox)


# ============================================================
#  ACTIONS
# ============================================================

class Actions:
    def __init__(self, region=None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.region = region  # (x, y, w, h)
        print(f"🎯 Screen: {self.screen_w}×{self.screen_h}")
        if region:
            print(f"🎯 Region: {region}")

    def img_to_screen(self, img_x, img_y, img_size):
        """Convert image pixel coords to screen pixel coords,
        accounting for game region."""
        img_w, img_h = img_size

        if self.region:
            rx, ry, rw, rh = self.region
            # Scale from image coords to region coords
            sx = rx + int(img_x / img_w * rw)
            sy = ry + int(img_y / img_h * rh)
        else:
            # Image is fullscreen
            sx = int(img_x / img_w * self.screen_w)
            sy = int(img_y / img_h * self.screen_h)

        # Clamp
        sx = max(0, min(sx, self.screen_w - 1))
        sy = max(0, min(sy, self.screen_h - 1))
        return sx, sy

    def move_and_click(self, img_x, img_y, img_size, duration=0.3):
        """Move cursor to a point (in image coords) and click."""
        sx, sy = self.img_to_screen(img_x, img_y, img_size)
        print(f"     🖱️  img({img_x},{img_y}) → screen({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=duration)
        time.sleep(0.1)
        pyautogui.click(sx, sy)
        return sx, sy

    def press(self, key="space"):
        print(f"     ⌨️  '{key}'")
        pyautogui.press(key)

    def click_screen(self, sx, sy, duration=0.25):
        """Click absolute screen coords."""
        print(f"     🖱️  screen({sx},{sy})")
        pyautogui.moveTo(sx, sy, duration=duration)
        time.sleep(0.1)
        pyautogui.click(sx, sy)


# ============================================================
#  GAME AI — hybrid OCR + LLM
# ============================================================

class GameAI:
    def __init__(self, vision_model="minicpm-v", game_region=None,
                 action_key="space", allowed_keys=None, debug=True, game_name="custom"):
        print("=" * 55)
        print("  🎮  AI Game Player — OCR + LLM Hybrid")
        print("=" * 55)

        self.screen_reader = ScreenReader(gpu=True)
        self.game_vision = GameVision(model=vision_model)
        self.actions = Actions(region=game_region)
        self.region = game_region
        self.action_key = action_key
        self.allowed_keys = allowed_keys  # None = single-key game
        self.game_rules = ""
        self.game_name = game_name
        self.debug = debug
        self.loop_count = 0
        self.stuck_count = 0
        self.last_state = "unknown"
        self.extra_button_keywords = []
        
        # LLM cooldown variables
        self.last_action_frame = 0
        self.action_cooldown = 3

        if debug:
            os.makedirs("debug_frames", exist_ok=True)

        print(f"\n   Action key: '{action_key}'")
        if allowed_keys:
            print(f"   Multi-key mode: {allowed_keys}")
        print(f"   Debug: {'ON' if debug else 'OFF'}")
        print()

    def set_game_rules(self, rules):
        self.game_rules = rules

    def add_button_keywords(self, keywords):
        """Add game-specific button keywords to look for."""
        self.extra_button_keywords.extend([k.lower() for k in keywords])

    def _save_debug(self, pil_img, tag):
        if self.debug:
            path = f"debug_frames/{self.loop_count:04d}_{tag}.png"
            pil_img.save(path)

    def _save_debug_annotated(self, pil_img, texts, tag):
        """Save screenshot with OCR results drawn on it."""
        if not self.debug:
            return
        cv_img = self.screen_reader.pil_to_cv(pil_img)
        for t in texts:
            x, y, w, h = t["x"], t["y"], t["w"], t["h"]
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{t['text']} ({t['confidence']:.2f})",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
        pil_out = self.screen_reader.cv_to_pil(cv_img)
        path = f"debug_frames/{self.loop_count:04d}_{tag}.png"
        pil_out.save(path)

    # ----------------------------------------------------------
    #  Handle non-playing states: find and click buttons
    # ----------------------------------------------------------
    def _handle_buttons(self, screenshot):
        """Find buttons using OCR and click the best one.
        This is FAST and PIXEL-PERFECT — no LLM involved."""

        # Try to find buttons by text
        buttons = self.screen_reader.find_buttons(
            screenshot, self.extra_button_keywords
        )

        if buttons:
            best = buttons[0]
            print(f"   🔘 Clicking button: \"{best['text']}\" "
                  f"(score={best['match_score']}, conf={best['confidence']:.2f})")
            self.actions.move_and_click(
                best["cx"], best["cy"], screenshot.size
            )
            self.stuck_count = 0
            return True

        # Fallback: try rectangle detection
        print("   📐 No text buttons found, trying rectangle detection...")
        target = self.screen_reader.find_clickable(screenshot)
        if target:
            cx, cy = target
            print(f"   📐 Clicking detected element at ({cx}, {cy})")
            self.actions.move_and_click(cx, cy, screenshot.size)
            self.stuck_count = 0
            return True

        # Nothing found
        self.stuck_count += 1
        print(f"   ⚠️  No clickable elements found (stuck={self.stuck_count})")

        if self.stuck_count >= 3:
            # Try pressing space or clicking center
            print("   ⚠️  Trying space key as fallback")
            self.actions.press("space")
            self.stuck_count = 0

        return False

    # ----------------------------------------------------------
    #  Handle playing state: ask LLM if we should act
    # ----------------------------------------------------------
    def _handle_playing(self, screenshot):
        """Ask the LLM whether to press a key. Only called during
        active gameplay — this is the ONE place the LLM is used."""

        # Cooldown check - don't spam LLM every frame
        if (self.loop_count - self.last_action_frame) < self.action_cooldown:
            return

        if self.allowed_keys:
            # Multi-key game (e.g., 2048, platformer)
            key, reason = self.game_vision.choose_key(
                screenshot, self.game_rules, self.allowed_keys
            )
            if key and key != "none":
                print(f"   🎮 Pressing '{key}' — {reason}")
                self.actions.press(key)
                self.last_action_frame = self.loop_count
            else:
                print(f"   🎮 Waiting — {reason}")
        else:
            # Single-key game (e.g., Flappy Bird, Dino)
            should, reason = self.game_vision.should_act(
                screenshot, self.game_rules
            )
            if should:
                print(f"   🎮 Acting! '{self.action_key}' — {reason}")
                self.actions.press(self.action_key)
                self.last_action_frame = self.loop_count
            else:
                print(f"   🎮 Waiting — {reason}")

    # ----------------------------------------------------------
    #  Main step
    # ----------------------------------------------------------
    def step(self):
        self.loop_count += 1
        print(f"\n{'─'*20}  Loop {self.loop_count}  {'─'*20}")

        # 1. CAPTURE
        t0 = time.time()
        screenshot = capture_screen(self.region)
        self._save_debug(screenshot, "raw")

        # 2. FAST STATE DETECTION (OCR-based, no LLM)
        state, all_text = self.screen_reader.detect_state_fast(screenshot, game_name=self.game_name)
        ocr_time = time.time() - t0
        print(f"   📖 OCR: {ocr_time:.2f}s | state={state} | "
              f"found {len(all_text)} text regions")

        # Show what OCR found
        if all_text:
            texts_preview = [f"\"{t['text']}\"" for t in all_text[:5]]
            print(f"   📖 Text: {', '.join(texts_preview)}")

        self._save_debug_annotated(screenshot, all_text, "ocr")

        # 3. ACT BASED ON STATE
        if state in ("game_over", "menu", "paused"):
            print(f"   📋 State: {state} → looking for buttons...")
            self._handle_buttons(screenshot)

        elif state == "playing":
            print(f"   📋 State: playing → asking LLM...")
            t0 = time.time()
            self._handle_playing(screenshot)
            llm_time = time.time() - t0
            print(f"   🧠 LLM decision: {llm_time:.1f}s")

        else:
            # Uncertain — use LLM to classify
            print(f"   📋 State uncertain, asking LLM...")
            llm_state = self.game_vision.classify_state(
                screenshot, self.game_rules
            )
            print(f"   🧠 LLM says: {llm_state}")

            if llm_state in ("game_over", "menu", "paused"):
                self._handle_buttons(screenshot)
            elif llm_state == "playing":
                self._handle_playing(screenshot)
            else:
                self.actions.press(self.action_key)

        self.last_state = state

    # ----------------------------------------------------------
    #  Run loop
    # ----------------------------------------------------------
    def run(self, max_loops=None, delay=0.3):
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
            print("\n⛔ Stopped by user")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe (mouse in corner)")

        print(f"\n📊 {loop} loops completed")


# ============================================================
#  PRESETS
# ============================================================

PRESETS = {
    "flappy_bird": {
        "rules": "Flappy Bird: the bird falls constantly. Press space to flap "
                 "upward. Avoid pipes. If a pipe is approaching and the bird "
                 "is level with or below the gap, press space. If bird is "
                 "above the gap, wait.",
        "action_key": "space",
        "allowed_keys": None,       # single-key game
        "extra_buttons": ["tap", "flap"],
    },
    "dino_run": {
        "rules": "Chrome Dino: press space to jump over cacti. Press down "
                 "to duck under birds. If an obstacle is close, act.",
        "action_key": "space",
        "allowed_keys": ["space", "down"],
        "extra_buttons": [],
    },
    "2048": {
        "rules": "2048: merge tiles by pressing arrow keys. Try to keep "
                 "the highest tile in a corner. Build a chain.",
        "action_key": "up",
        "allowed_keys": ["up", "down", "left", "right"],
        "extra_buttons": ["new game", "new"],
    },
    "cookie_clicker": {
        "rules": "Click the big cookie as fast as possible.",
        "action_key": "space",
        "allowed_keys": None,
        "extra_buttons": ["cookie"],
    },
}


# ============================================================
#  REGION PICKER
# ============================================================

def pick_region():
    print("\n🖱️  Move mouse to TOP-LEFT of game window, press Enter")
    input("   → ")
    x1, y1 = pyautogui.position()
    print(f"   Got: ({x1}, {y1})")

    print("   Move mouse to BOTTOM-RIGHT of game window, press Enter")
    input("   → ")
    x2, y2 = pyautogui.position()
    print(f"   Got: ({x2}, {y2})")

    region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    print(f"   ✅ Region: {region}")
    return region


# ============================================================
#  QUICK TEST — run this to verify OCR works on your screen
# ============================================================

def test_ocr(region=None):
    """Capture screen, run OCR, show what was found."""
    print("\n🧪 Testing OCR...\n")
    reader = ScreenReader(gpu=True)
    screenshot = capture_screen(region)

    print(f"   Screenshot size: {screenshot.size}")
    print(f"\n   All text found:")

    texts = reader.find_all_text(screenshot)
    for t in texts:
        print(f"      \"{t['text']}\" at ({t['cx']},{t['cy']}) "
              f"conf={t['confidence']:.2f}")

    print(f"\n   Buttons found:")
    buttons = reader.find_buttons(screenshot)
    for b in buttons:
        print(f"      \"{b['text']}\" at ({b['cx']},{b['cy']}) "
              f"matched='{b['matched_keyword']}' score={b['match_score']}")

    print(f"\n   Rectangles found:")
    rects = reader.find_rectangles(screenshot)
    for r in rects[:5]:
        print(f"      ({r['cx']},{r['cy']}) size={r['w']}x{r['h']}")

    state, _ = reader.detect_state_fast(screenshot)
    print(f"\n   Detected state: {state}")

    # Save annotated image
    cv_img = reader.pil_to_cv(screenshot)
    for t in texts:
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(cv_img, t["text"], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for b in buttons:
        cv2.circle(cv_img, (b["cx"], b["cy"]), 10, (0, 0, 255), -1)
    out = reader.cv_to_pil(cv_img)
    out.save("debug_ocr_test.png")
    print(f"\n   Saved: debug_ocr_test.png")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n🎮 AI Game Player — Setup\n")
    print("Modes:")
    print("  1. Play a game")
    print("  2. Test OCR on current screen")

    mode = input("\nChoice [1]: ").strip() or "1"

    if mode == "2":
        reg_choice = input("Region? (p)ick / (f)ullscreen [f]: ").strip().lower()
        region = pick_region() if reg_choice == "p" else None
        test_ocr(region)
        return

    # ---- Game selection ----
    presets = list(PRESETS.keys())
    for i, name in enumerate(presets, 1):
        print(f"  {i}. {name}")
    print(f"  {len(presets) + 1}. custom")

    choice = input("\nGame: ").strip()
    preset = None
    selected_game_name = "custom"

    if choice.isdigit() and int(choice) <= len(presets):
        key = presets[int(choice) - 1]
        selected_game_name = key
        preset = PRESETS[key]
        print(f"✅ {key}")

    if preset:
        rules = preset["rules"]
        action_key = preset["action_key"]
        allowed_keys = preset["allowed_keys"]
        extra_buttons = preset.get("extra_buttons", [])
    else:
        rules = input("Game rules: ").strip()
        action_key = input("Action key [space]: ").strip() or "space"
        mk = input("Multiple keys? comma-separated or empty: ").strip()
        allowed_keys = [k.strip() for k in mk.split(",")] if mk else None
        extra_buttons = []
        eb = input("Extra button keywords (comma-sep, or empty): ").strip()
        if eb:
            extra_buttons = [k.strip() for k in eb.split(",")]

    # ---- Vision model ----
    v_model = input("Vision model [minicpm-v]: ").strip() or "minicpm-v"

    # ---- Timing ----
    delay = float(input("Loop delay [0.3]: ").strip() or "0.3")
    max_l = input("Max loops [unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None

    # ---- Region ----
    reg = input("Region? (m)anual / (p)ick / (f)ullscreen [f]: ").strip().lower()
    region = None
    if reg == "m":
        try:
            region = tuple(int(p) for p in input("  x,y,w,h: ").split(","))
        except ValueError:
            print("  Bad format, using fullscreen")
    elif reg == "p":
        region = pick_region()

    debug = input("Save debug frames? (y/n) [y]: ").strip().lower() != "n"

    # ---- Build ----
    ai = GameAI(
        vision_model=v_model,
        game_region=region,
        action_key=action_key,
        allowed_keys=allowed_keys,
        debug=debug,
        game_name=selected_game_name,
    )
    ai.set_game_rules(rules)
    if extra_buttons:
        ai.add_button_keywords(extra_buttons)

    # ---- Run ----
    run_mode = input("(r)un / (s)tep [r]: ").strip().lower()
    if run_mode == "s":
        while True:
            ai.step()
            if input("Next? (Enter/q): ").strip().lower() == "q":
                break
    else:
        ai.run(max_loops=max_loops, delay=delay)


if __name__ == "__main__":
    main()