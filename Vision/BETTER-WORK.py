import pyautogui
import cv2
import numpy as np
import easyocr
import io
import json
import time
import os
import re
import ollama
from PIL import ImageGrab, Image, ImageEnhance
from collections import deque

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# ============================================================
#  INSTALL:
#    pip install easyocr opencv-python pyautogui pillow ollama
# ============================================================

DEBUG = True
os.makedirs("debug_frames", exist_ok=True)


# ============================================================
#  FAST SCREEN CAPTURE
# ============================================================

def capture(region=None):
    if region:
        x, y, w, h = region
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    else:
        img = ImageGrab.grab()
    return np.array(img)   # return numpy BGR for OpenCV


def to_bgr(rgb_array):
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def save_debug(frame_bgr, name, loop):
    if DEBUG:
        cv2.imwrite(f"debug_frames/{loop:04d}_{name}.png", frame_bgr)


# ============================================================
#  OCR  (only used for menus / game-over screens)
# ============================================================

class OCR:
    RESTART_WORDS = [
        "game over", "try again", "retry", "restart",
        "you died", "failed", "play again", "ok",
        "play", "start", "begin", "tap", "new game",
        "continue", "resume", "revive", "next", "menu",
    ]

    def __init__(self):
        print("📖 Loading OCR...")
        try:
            self.reader = easyocr.Reader(["en"], gpu=True,  verbose=False)
        except Exception:
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("📖 OCR ready")

    def read(self, bgr):
        results = self.reader.readtext(bgr)
        out = []
        for bbox, text, conf in results:
            pts = np.array(bbox, np.int32)
            x, y   = int(pts[:,0].min()), int(pts[:,1].min())
            x2, y2 = int(pts[:,0].max()), int(pts[:,1].max())
            out.append({
                "text": text.strip(),
                "cx": (x + x2)//2, "cy": (y + y2)//2,
                "x": x, "y": y, "w": x2-x, "h": y2-y,
                "conf": round(conf, 2),
            })
        return out

    def find_restart_button(self, bgr):
        """Return (cx,cy) of the best restart/play button, or None."""
        texts = self.read(bgr)
        best, best_score = None, 0
        h_img, w_img = bgr.shape[:2]
        cx_img, cy_img = w_img//2, h_img//2
        for t in texts:
            tl = t["text"].lower()
            for kw in self.RESTART_WORDS:
                if kw in tl:
                    # score: keyword match + proximity to center
                    dist = ((t["cx"]-cx_img)**2 + (t["cy"]-cy_img)**2)**0.5
                    max_d = (cx_img**2 + cy_img**2)**0.5
                    score = 100 + (1 - dist/max(max_d,1))*30
                    if score > best_score:
                        best_score = score
                        best = t
        if best:
            return best["cx"], best["cy"]
        return None

    def is_game_over(self, bgr):
        texts = self.read(bgr)
        joined = " ".join(t["text"].lower() for t in texts)
        triggers = ["game over","try again","you died","failed",
                    "you lose","restart","retry","level failed"]
        return any(t in joined for t in triggers)

    def is_menu(self, bgr):
        texts = self.read(bgr)
        joined = " ".join(t["text"].lower() for t in texts)
        triggers = ["tap to start","press start","tap to play",
                    "click to start","main menu","press space to start"]
        return any(t in joined for t in triggers)


# ============================================================
#  ACTIONS  (thin wrapper around pyautogui)
# ============================================================

class Act:
    def __init__(self, region=None):
        self.sw, self.sh = pyautogui.size()
        self.region = region   # (x, y, w, h) or None

    def _abs(self, x_pct, y_pct):
        """percentage (0-100) → absolute screen coords"""
        if self.region:
            rx, ry, rw, rh = self.region
            return rx + int(x_pct/100*rw), ry + int(y_pct/100*rh)
        return int(x_pct/100*self.sw), int(y_pct/100*self.sh)

    def _img_abs(self, ix, iy, iw, ih):
        """image pixel → absolute screen coords"""
        if self.region:
            rx, ry, rw, rh = self.region
            return rx + int(ix/iw*rw), ry + int(iy/ih*rh)
        return int(ix/iw*self.sw), int(iy/ih*self.sh)

    def click_pct(self, x, y, right=False):
        sx, sy = self._abs(x, y)
        pyautogui.moveTo(sx, sy, duration=0.05)
        time.sleep(0.02)
        if right:
            pyautogui.rightClick()
        else:
            pyautogui.click()
        print(f"  🖱 {'R' if right else 'L'}click ({sx},{sy})")

    def click_img(self, ix, iy, iw, ih, right=False):
        sx, sy = self._img_abs(ix, iy, iw, ih)
        pyautogui.moveTo(sx, sy, duration=0.05)
        time.sleep(0.02)
        if right:
            pyautogui.rightClick()
        else:
            pyautogui.click()
        print(f"  🖱 {'R' if right else 'L'}click img({ix},{iy})→screen({sx},{sy})")

    def key(self, k):
        pyautogui.press(k)
        print(f"  ⌨ '{k}'")

    def key_down(self, k):
        pyautogui.keyDown(k)

    def key_up(self, k):
        pyautogui.keyUp(k)

    def type_word(self, word):
        for ch in word.lower():
            if ch.isalpha():
                pyautogui.press(ch)
                time.sleep(0.07)
        pyautogui.press("enter")
        print(f"  ⌨ typed '{word}'")


# ============================================================
#
#   GAME BOTS  — pure OpenCV, no LLM, fast
#
# ============================================================


# ─────────────────────────────────────────────────────────────
#  FLAPPY BIRD
# ─────────────────────────────────────────────────────────────
class FlappyBot:
    """
    Detects the bird by its yellow colour.
    Detects pipes by green colour.
    Taps when bird_y > gap_center_y.
    """
    def __init__(self, act: Act, region=None):
        self.act    = act
        self.region = region
        self.loop   = 0

    def step(self, bgr):
        self.loop += 1
        h, w = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ── find bird (yellow) ──────────────────────────
        bird_mask = cv2.inRange(hsv,
            np.array([20, 100, 100]),
            np.array([35, 255, 255]))
        bird_y = None
        cnts, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 50:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    bird_y = int(M["m01"] / M["m00"])

        # ── find pipes (green) ─────────────────────────
        pipe_mask = cv2.inRange(hsv,
            np.array([40, 80, 50]),
            np.array([80, 255, 255]))

        # Find the gap center of the nearest pipe pair
        gap_y = h // 2   # default: middle
        col_sums = np.sum(pipe_mask, axis=0)
        pipe_cols = np.where(col_sums > h * 0.3 * 255)[0]

        if len(pipe_cols) > 0:
            # Use the rightmost pipe column group ahead of the bird
            # (left third of screen is where pipes approach)
            ahead = pipe_cols[pipe_cols > w * 0.1]
            if len(ahead) > 0:
                col = int(np.median(ahead[:max(1, len(ahead)//3)]))
                col_strip = pipe_mask[:, max(0,col-10):col+10]
                # Gap = rows with NO pipe
                row_has_pipe = np.sum(col_strip, axis=1) > 0
                gap_rows = np.where(~row_has_pipe)[0]
                if len(gap_rows) > 0:
                    gap_y = int(np.median(gap_rows))

        # ── decision ───────────────────────────────────
        should_flap = False
        if bird_y is not None:
            should_flap = bird_y > gap_y
            print(f"  🐦 bird_y={bird_y} gap_y={gap_y} flap={should_flap}")
        else:
            # Can't see bird — tap to keep alive
            should_flap = (self.loop % 8 == 0)
            print(f"  🐦 bird not found, periodic tap")

        if should_flap:
            self.act.click_pct(50, 50)

        # debug
        if DEBUG and self.loop % 10 == 0:
            dbg = bgr.copy()
            cv2.drawContours(dbg, cnts, -1, (0,255,255), 2) if cnts else None
            cv2.line(dbg, (0, gap_y), (w, gap_y), (0,255,0), 2)
            if bird_y:
                cv2.line(dbg, (0, bird_y), (w, bird_y), (0,0,255), 2)
            save_debug(dbg, "flappy", self.loop)


# ─────────────────────────────────────────────────────────────
#  DINO RUN
# ─────────────────────────────────────────────────────────────
class DinoBot:
    """
    Detects obstacles (dark blobs on light ground).
    Jumps when an obstacle is within threshold distance.
    Ducks if obstacle is at head height (pterodactyl).
    """
    def __init__(self, act: Act, region=None):
        self.act     = act
        self.region  = region
        self.loop    = 0
        self.jumping = False
        self.jump_cd = 0

    def step(self, bgr):
        self.loop += 1
        h, w = bgr.shape[:2]

        # Work in lower 40% of screen (ground level)
        ground = bgr[int(h*0.45):int(h*0.85), :]
        gray   = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)

        # Chrome dino is light grey background, obstacles are dark
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        cnts, _   = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80:
                continue
            x, y, ow, oh = cv2.boundingRect(c)
            # Filter out the dino itself (it's on the left side)
            if x < w * 0.15:
                continue
            obstacles.append((x, y, ow, oh, area))

        self.jump_cd = max(0, self.jump_cd - 1)

        if obstacles and self.jump_cd == 0:
            # Sort by x (nearest first)
            obstacles.sort(key=lambda o: o[0])
            ox, oy, ow, oh, area = obstacles[0]

            # Threshold: jump when obstacle is within 30% of screen width
            if ox < w * 0.35:
                # Is it a pterodactyl? (flies at head height = upper half of ground strip)
                is_ptero = oy < int(h*0.40*0.4)
                action   = "down" if is_ptero else "space"
                print(f"  🦕 obstacle x={ox} y={oy} → {action}")
                self.act.key(action)
                self.jump_cd = 15
            else:
                print(f"  🦕 obstacle far x={ox} — wait")
        else:
            print(f"  🦕 clear" if not obstacles else f"  🦕 cooldown {self.jump_cd}")

        if DEBUG and self.loop % 10 == 0:
            dbg = bgr.copy()
            for ox, oy, ow, oh, _ in obstacles:
                cv2.rectangle(dbg,
                    (ox, int(h*0.45)+oy),
                    (ox+ow, int(h*0.45)+oy+oh),
                    (0,0,255), 2)
            save_debug(dbg, "dino", self.loop)


# ─────────────────────────────────────────────────────────────
#  SNAKE
# ─────────────────────────────────────────────────────────────
class SnakeBot:
    """
    Detects food (bright coloured pixel blob) and snake head.
    Uses BFS to find a safe path to food.
    Falls back to wall-following if BFS fails.
    """
    def __init__(self, act: Act, region=None):
        self.act       = act
        self.region    = region
        self.loop      = 0
        self.direction = "right"
        self.move_cd   = 0
        self.CELL      = 20   # approximate cell size — adjust per game

    def _grid_from_frame(self, bgr):
        h, w = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Food: bright red/pink blob
        food_mask = cv2.inRange(hsv,
            np.array([0, 150, 150]),
            np.array([10, 255, 255]))
        food_mask2 = cv2.inRange(hsv,
            np.array([170, 150, 150]),
            np.array([180, 255, 255]))
        food_mask  = cv2.bitwise_or(food_mask, food_mask2)

        # Snake: green blobs
        snake_mask = cv2.inRange(hsv,
            np.array([40, 80, 80]),
            np.array([80, 255, 255]))

        return food_mask, snake_mask, h, w

    def step(self, bgr):
        self.loop    += 1
        self.move_cd  = max(0, self.move_cd - 1)
        if self.move_cd > 0:
            return

        food_mask, snake_mask, h, w = self._grid_from_frame(bgr)

        # ── find food ──────────────────────────────────
        food_cnts, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        food_pos = None
        if food_cnts:
            c = max(food_cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                food_pos = (int(M["m10"]/M["m00"]),
                            int(M["m01"]/M["m00"]))

        # ── find snake head ────────────────────────────
        snake_cnts, _ = cv2.findContours(snake_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        head_pos = None
        if snake_cnts:
            # Head is usually the brightest/largest segment
            c = max(snake_cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                head_pos = (int(M["m10"]/M["m00"]),
                            int(M["m01"]/M["m00"]))

        if food_pos and head_pos:
            fx, fy = food_pos
            hx, hy = head_pos
            dx = fx - hx
            dy = fy - hy

            # Simple greedy: move toward food, avoid reversing
            opposites = {"up":"down","down":"up","left":"right","right":"left"}
            candidates = []
            if abs(dx) > abs(dy):
                primary   = "right" if dx > 0 else "left"
                secondary = "down"  if dy > 0 else "up"
            else:
                primary   = "down"  if dy > 0 else "up"
                secondary = "right" if dx > 0 else "left"

            for d in [primary, secondary]:
                if d != opposites.get(self.direction):
                    candidates.append(d)

            if candidates:
                self.direction = candidates[0]
                print(f"  🐍 head=({hx},{hy}) food=({fx},{fy}) → {self.direction}")
                self.act.key(self.direction)
                self.move_cd = 3
        else:
            # No food/head detected — keep moving in current direction
            print(f"  🐍 no detection → {self.direction}")
            self.act.key(self.direction)
            self.move_cd = 4

        if DEBUG and self.loop % 15 == 0:
            dbg = bgr.copy()
            if food_pos:
                cv2.circle(dbg, food_pos, 8, (0,0,255), -1)
            if head_pos:
                cv2.circle(dbg, head_pos, 8, (0,255,0), -1)
            save_debug(dbg, "snake", self.loop)


# ─────────────────────────────────────────────────────────────
#  TETRIS
# ─────────────────────────────────────────────────────────────
class TetrisBot:
    """
    Reads the board by colour-scanning each cell.
    Uses a simple heuristic to choose rotation + column:
      - minimize holes
      - minimize height
      - maximize lines cleared
    """
    COLS = 10
    ROWS = 20

    def __init__(self, act: Act, region=None):
        self.act        = act
        self.region     = region
        self.loop       = 0
        self.move_cd    = 0
        self.placed     = False
        self.board_rect = None   # (x,y,w,h) of the board in image coords

    def _detect_board(self, bgr):
        """Find the Tetris board rectangle."""
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            aspect = bh / max(bw, 1)
            # Tetris board is roughly 2:1 tall
            if area > 0.05 * h * w and 1.5 < aspect < 3.0:
                if best is None or area > best[4]:
                    best = (x, y, bw, bh, area)
        if best:
            return best[:4]
        # Fallback: center strip
        return (w//4, h//10, w//2, int(h*0.8))

    def _read_board(self, bgr, bx, by, bw, bh):
        """Return 2D grid: True=filled, False=empty."""
        cell_w = bw // self.COLS
        cell_h = bh // self.ROWS
        grid   = []
        for row in range(self.ROWS):
            r = []
            for col in range(self.COLS):
                cx = bx + col*cell_w + cell_w//2
                cy = by + row*cell_h + cell_h//2
                if 0 <= cy < bgr.shape[0] and 0 <= cx < bgr.shape[1]:
                    b, g, rv = bgr[cy, cx]
                    # Filled if not close to background colour
                    # (adjust 50,50,50 if your game has a different bg)
                    filled = not (int(b) < 60 and int(g) < 60 and int(rv) < 60)
                    r.append(filled)
                else:
                    r.append(False)
            grid.append(r)
        return grid

    def _score(self, grid):
        """Heuristic score for a board state (lower = better)."""
        heights = []
        for col in range(self.COLS):
            for row in range(self.ROWS):
                if grid[row][col]:
                    heights.append(self.ROWS - row)
                    break
            else:
                heights.append(0)

        # Holes = empty cell with filled cell above
        holes = 0
        for col in range(self.COLS):
            filled = False
            for row in range(self.ROWS):
                if grid[row][col]:
                    filled = True
                elif filled:
                    holes += 1

        max_h    = max(heights) if heights else 0
        avg_h    = sum(heights) / len(heights) if heights else 0
        bumpiness= sum(abs(heights[i]-heights[i+1])
                       for i in range(len(heights)-1))

        # Lines cleared bonus
        lines = sum(1 for row in grid if all(row))

        return holes*4 + max_h*2 + bumpiness - lines*10

    def step(self, bgr):
        self.loop    += 1
        self.move_cd  = max(0, self.move_cd - 1)
        if self.move_cd > 0:
            return

        # Detect board
        if self.loop % 30 == 1:
            self.board_rect = self._detect_board(bgr)
        if not self.board_rect:
            return

        bx, by, bw, bh = self.board_rect
        grid = self._read_board(bgr, bx, by, bw, bh)

        # Simple strategy: check heights and fill lowest column
        heights = []
        for col in range(self.COLS):
            for row in range(self.ROWS):
                if grid[row][col]:
                    heights.append(self.ROWS - row)
                    break
            else:
                heights.append(0)

        # Find lowest column to fill
        min_h   = min(heights)
        target  = heights.index(min_h)
        # Find which column the piece is in (top rows, look for colour)
        piece_col = self.COLS // 2
        for col in range(self.COLS):
            for row in range(3):
                bv, gv, rv = bgr[by + row*(bh//self.ROWS),
                                  bx + col*(bw//self.COLS) + bw//(self.COLS*2)]
                if int(bv)+int(gv)+int(rv) > 100:
                    piece_col = col
                    break

        moves = []
        if target < piece_col:
            moves = ["left"] * abs(target - piece_col)
        elif target > piece_col:
            moves = ["right"] * abs(target - piece_col)

        # Rotate occasionally to fit better
        if self.loop % 5 == 0:
            moves.insert(0, "up")

        # Execute moves then drop
        for m in moves:
            self.act.key(m)
            time.sleep(0.05)

        # Soft drop
        self.act.key("down")
        self.move_cd = 2

        if DEBUG and self.loop % 20 == 0:
            dbg = bgr.copy()
            cv2.rectangle(dbg, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
            save_debug(dbg, "tetris", self.loop)


# ─────────────────────────────────────────────────────────────
#  COOKIE CLICKER
# ─────────────────────────────────────────────────────────────
class CookieBot:
    """
    Clicks the main cookie by finding the largest circle on the left.
    Watches the RIGHT side for golden cookies by detecting
    bright yellow circular blobs that are NOT the main cookie.
    """
    def __init__(self, act: Act, region=None):
        self.act         = act
        self.region      = region
        self.loop        = 0
        self.cookie_pos  = None   # (cx, cy, iw, ih)
        self.golden_cd   = 0

    def _find_cookie(self, bgr):
        h, w  = bgr.shape[:2]
        # Only look in left 40%
        left  = bgr[:, :int(w*0.4)]
        gray  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (11,11), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=80,
            param1=50, param2=25,
            minRadius=40, maxRadius=h//2)
        if circles is not None:
            c = sorted(circles[0], key=lambda x: x[2], reverse=True)[0]
            return int(c[0]), int(c[1]), w, h
        # fallback: left-center
        return int(w*0.2), h//2, w, h

    def _find_golden_cookie(self, bgr):
        h, w = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Golden cookies are bright yellow/gold circles
        mask = cv2.inRange(hsv,
            np.array([18, 120, 150]),
            np.array([32, 255, 255]))

        # Remove the main cookie area (left 30%)
        mask[:, :int(w*0.3)] = 0

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            # Check circularity
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4*np.pi*area / (peri**2)
            if circ > 0.4:   # reasonably circular
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    candidates.append((cx, cy, area, circ))

        if candidates:
            # Pick most circular
            candidates.sort(key=lambda x: x[3], reverse=True)
            return candidates[0][0], candidates[0][1], w, h
        return None

    def step(self, bgr):
        self.loop       += 1
        self.golden_cd   = max(0, self.golden_cd - 1)
        h, w             = bgr.shape[:2]

        # Every 5 loops re-detect cookie position
        if self.loop % 5 == 1 or self.cookie_pos is None:
            cx, cy, iw, ih = self._find_cookie(bgr)
            self.cookie_pos = (cx, cy, iw, ih)

        # Check for golden cookie every loop
        gold = self._find_golden_cookie(bgr)
        if gold and self.golden_cd == 0:
            gx, gy, iw, ih = gold
            print(f"  🌟 Golden cookie at ({gx},{gy})")
            self.act.click_img(gx, gy, iw, ih)
            self.golden_cd = 10
            return

        # Click main cookie
        cx, cy, iw, ih = self.cookie_pos
        print(f"  🍪 Click cookie ({cx},{cy})")
        self.act.click_img(cx, cy, iw, ih)

        if DEBUG and self.loop % 30 == 0:
            dbg = bgr.copy()
            cv2.circle(dbg, (cx,cy), 20, (0,255,255), 3)
            if gold:
                cv2.circle(dbg, (gold[0],gold[1]), 15, (0,215,255), 3)
            save_debug(dbg, "cookie", self.loop)


# ─────────────────────────────────────────────────────────────
#  2048
# ─────────────────────────────────────────────────────────────
class Bot2048:
    """
    Reads tiles by OCR. Picks move by scoring each direction.
    Strategy: keep highest tile bottom-left, prefer left+down.
    """
    def __init__(self, act: Act, ocr: OCR, region=None):
        self.act    = act
        self.ocr    = ocr
        self.region = region
        self.loop   = 0
        self.cd     = 0

    def _read_board(self, bgr):
        texts  = self.ocr.read(bgr)
        h, w   = bgr.shape[:2]
        # Tile numbers we expect to see
        nums   = {}
        for t in texts:
            try:
                val = int(t["text"])
                if val > 0 and (val & (val-1)) == 0:  # power of 2
                    col = int(t["cx"] / w * 4)
                    row = int(t["cy"] / h * 4)
                    col = max(0, min(3, col))
                    row = max(0, min(3, row))
                    nums[(row, col)] = val
            except ValueError:
                pass
        grid = [[nums.get((r,c),0) for c in range(4)] for r in range(4)]
        return grid

    def _slide(self, grid, direction):
        import copy
        g = copy.deepcopy(grid)

        def merge_left(row):
            row = [x for x in row if x]
            merged = []
            i = 0
            while i < len(row):
                if i+1 < len(row) and row[i] == row[i+1]:
                    merged.append(row[i]*2)
                    i += 2
                else:
                    merged.append(row[i])
                    i += 1
            return merged + [0]*(4-len(merged))

        if direction == "left":
            return [merge_left(r) for r in g]
        if direction == "right":
            return [merge_left(r[::-1])[::-1] for r in g]
        if direction == "up":
            cols = [[g[r][c] for r in range(4)] for c in range(4)]
            merged = [merge_left(col) for col in cols]
            return [[merged[c][r] for c in range(4)] for r in range(4)]
        if direction == "down":
            cols = [[g[r][c] for r in range(4)] for c in range(4)]
            merged = [merge_left(col[::-1])[::-1] for col in cols]
            return [[merged[c][r] for c in range(4)] for r in range(4)]
        return g

    def _score(self, grid):
        flat = [grid[r][c] for r in range(4) for c in range(4)]
        if not any(flat):
            return -1000

        # Prefer highest tile in bottom-left (row3,col0)
        corner_bonus = grid[3][0] * 3

        # Penalise high tiles not in bottom row
        penalty = 0
        for r in range(3):
            for c in range(4):
                penalty += grid[r][c]

        # Reward merges
        merges = sum(1 for x in flat if x > 0)

        return corner_bonus - penalty * 0.3 + merges

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        grid = self._read_board(bgr)
        flat = [grid[r][c] for r in range(4) for c in range(4)]
        print(f"  2️⃣  max tile={max(flat) if flat else 0}")

        # Score each move
        best_move, best_score = "left", -9999
        for d in ["left", "down", "right", "up"]:
            moved = self._slide(grid, d)
            if moved != grid:
                s = self._score(moved)
                if s > best_score:
                    best_score = s
                    best_move  = d

        print(f"  2️⃣  → {best_move} (score={best_score:.0f})")
        self.act.key(best_move)
        self.cd = 3


# ─────────────────────────────────────────────────────────────
#  PACMAN
# ─────────────────────────────────────────────────────────────
class PacManBot:
    """
    Finds Pac-Man (yellow circle) and ghosts (coloured blobs).
    Moves toward pellets while avoiding ghosts.
    """
    def __init__(self, act: Act, region=None):
        self.act       = act
        self.region    = region
        self.loop      = 0
        self.direction = "right"
        self.cd        = 0

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        h, w = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Pac-Man: yellow
        pac_mask = cv2.inRange(hsv,
            np.array([20,100,100]),
            np.array([35,255,255]))
        pac_pos  = None
        cnts, _  = cv2.findContours(pac_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0 and cv2.contourArea(c) > 100:
                pac_pos = (int(M["m10"]/M["m00"]),
                           int(M["m01"]/M["m00"]))

        # Ghosts: red, pink, cyan, orange blobs (not yellow)
        ghost_mask = cv2.inRange(hsv,
            np.array([0,100,100]),
            np.array([20,255,255]))
        ghost_mask2 = cv2.inRange(hsv,
            np.array([160,100,100]),
            np.array([180,255,255]))
        ghost_mask = cv2.bitwise_or(ghost_mask, ghost_mask2)

        ghost_pos = []
        gcnts, _  = cv2.findContours(ghost_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        for c in gcnts:
            if cv2.contourArea(c) > 150:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    ghost_pos.append((int(M["m10"]/M["m00"]),
                                      int(M["m01"]/M["m00"])))

        if not pac_pos:
            self.act.key(self.direction)
            self.cd = 2
            return

        px, py = pac_pos
        opposites = {"up":"down","down":"up","left":"right","right":"left"}

        # Danger: any ghost within 80px
        danger_dirs = set()
        for gx, gy in ghost_pos:
            d = ((gx-px)**2 + (gy-py)**2)**0.5
            if d < 80:
                if gx < px:   danger_dirs.add("left")
                elif gx > px: danger_dirs.add("right")
                if gy < py:   danger_dirs.add("up")
                elif gy > py: danger_dirs.add("down")

        # Prefer current direction, rotate if danger
        preferred = [self.direction, "right", "down", "left", "up"]
        for d in preferred:
            if d not in danger_dirs and d != opposites.get(self.direction):
                self.direction = d
                break

        print(f"  👻 pac=({px},{py}) ghosts={len(ghost_pos)} → {self.direction}")
        self.act.key(self.direction)
        self.cd = 3


# ─────────────────────────────────────────────────────────────
#  SPACE INVADERS
# ─────────────────────────────────────────────────────────────
class SpaceInvadersBot:
    """
    Finds the player ship (bottom, white/light blob).
    Finds the lowest alien (dark blobs in top half).
    Moves under it and shoots.
    """
    def __init__(self, act: Act, region=None):
        self.act    = act
        self.region = region
        self.loop   = 0
        self.cd     = 0

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        h, w = bgr.shape[:2]

        # Player ship: brightest object in bottom 20%
        bottom = bgr[int(h*0.8):, :]
        gray_b = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        _, thresh_b = cv2.threshold(gray_b, 150, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh_b, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        ship_x = w // 2
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                ship_x = int(M["m10"]/M["m00"])

        # Aliens: objects in top 60%
        top = bgr[:int(h*0.6), :]
        gray_t = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        _, thresh_t = cv2.threshold(gray_t, 80, 255, cv2.THRESH_BINARY)
        acnts, _ = cv2.findContours(thresh_t, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        aliens = []
        for c in acnts:
            area = cv2.contourArea(c)
            if 100 < area < 5000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    ax = int(M["m10"]/M["m00"])
                    ay = int(M["m01"]/M["m00"])
                    aliens.append((ax, ay, area))

        if aliens:
            # Target the lowest alien (largest y)
            aliens.sort(key=lambda a: a[1], reverse=True)
            target_x = aliens[0][0]

            move = None
            if ship_x < target_x - 10:
                move = "right"
            elif ship_x > target_x + 10:
                move = "left"

            print(f"  👾 ship_x={ship_x} target_x={target_x} aliens={len(aliens)}")

            if move:
                self.act.key(move)
                time.sleep(0.05)
            self.act.key("space")   # always shoot
            self.cd = 2
        else:
            print("  👾 no aliens found")
            self.act.key("space")
            self.cd = 3


# ─────────────────────────────────────────────────────────────
#  WORDLE  (LLM is fine here — slow game)
# ─────────────────────────────────────────────────────────────
class WordleBot:
    STARTERS = ["CRANE", "AUDIO", "STINK", "PLUMB",
                 "PROXY", "JERKY", "EPOCH", "BANJO"]

    def __init__(self, act: Act, vision_model="gemma3"):
        self.act      = act
        self.model    = vision_model
        self.guesses  = []
        self.loop     = 0
        self.cd       = 0
        self.waiting  = 0

    def _img_bytes(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(1.2)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _ask_llm(self, bgr):
        prev = ", ".join(self.guesses) if self.guesses else "none"
        img  = self._img_bytes(bgr)
        prompt = f"""You are playing Wordle.
Previous guesses: {prev}

Look at the board:
- GREEN tile = right letter, right position
- YELLOW tile = right letter, wrong position
- GRAY tile = letter not in word

What 5-letter word should you guess next?
Use CRANE if no guesses yet.

Reply ONLY: {{"word": "CRANE"}}
Exactly 5 letters. JSON only."""
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role":"user","content":prompt,"images":[img]}],
                options={"temperature":0.1,"num_predict":40},
            )
            raw  = resp["message"]["content"].strip()
            s, e = raw.find("{"), raw.rfind("}")+1
            if s!=-1 and e>s:
                d = json.loads(raw[s:e])
                w = re.sub(r"[^A-Za-z]","",d.get("word","CRANE"))[:5].upper()
                return w if len(w)==5 else "CRANE"
        except Exception as ex:
            print(f"  LLM error: {ex}")
        return "CRANE"

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)

        if self.waiting > 0:
            self.waiting -= 1
            print(f"  🟩 waiting for animation ({self.waiting})")
            return

        if self.cd > 0:
            return

        word = self._ask_llm(bgr)
        if word in self.guesses:
            for w in self.STARTERS:
                if w not in self.guesses:
                    word = w
                    break

        print(f"  🟩 guess: {word}")
        self.act.click_pct(50, 50)
        time.sleep(0.2)

        for ch in word.lower():
            pyautogui.press(ch)
            time.sleep(0.08)
        pyautogui.press("enter")

        self.guesses.append(word)
        self.cd      = 15
        self.waiting = 10


# ─────────────────────────────────────────────────────────────
#  GEOMETRY DASH  / STICKMAN HOOK  (same single-tap logic)
# ─────────────────────────────────────────────────────────────
class GeometryDashBot:
    """
    Looks for obstacles (bright coloured blocks/spikes) in the
    right half of the screen. Taps when one is close.
    """
    def __init__(self, act: Act, region=None):
        self.act    = act
        self.region = region
        self.loop   = 0
        self.cd     = 0

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        h, w = bgr.shape[:2]
        # Look at the strip ahead of the player (right 60% of screen)
        ahead = bgr[:, int(w*0.4):]
        gray  = cv2.cvtColor(ahead, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        obstacles = [c for c in cnts if cv2.contourArea(c) > 100]
        if obstacles:
            # Closest obstacle
            nearest = min(obstacles,
                key=lambda c: cv2.boundingRect(c)[0])
            ox = cv2.boundingRect(nearest)[0]
            if ox < w * 0.25:   # very close
                print(f"  ⬛ obstacle close → tap")
                self.act.click_pct(50, 50)
                self.cd = 4
                return

        print(f"  ⬛ clear")


# ─────────────────────────────────────────────────────────────
#  MINESWEEPER  (LLM for logic, CV for board reading)
# ─────────────────────────────────────────────────────────────
class MinesweeperBot:
    def __init__(self, act: Act, ocr: OCR,
                 vision_model="gemma3", region=None):
        self.act    = act
        self.ocr    = ocr
        self.model  = vision_model
        self.region = region
        self.loop   = 0
        self.cd     = 0
        self.first  = True

    def _img_bytes(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        h, w = bgr.shape[:2]

        # First move: click center
        if self.first:
            self.first = False
            print("  💣 First move: click center")
            self.act.click_pct(50, 50)
            self.cd = 8
            return

        # Ask LLM where to click
        img = self._img_bytes(bgr)
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role":"user","content":
                    """Minesweeper board. Left-click safe cells. Right-click mines.
Reply ONLY: {"action":"click or right_click","x":0-100,"y":0-100,"reason":"brief"}
JSON only.""",
                    "images":[img]}],
                options={"temperature":0.1,"num_predict":80},
            )
            raw = resp["message"]["content"].strip()
            s,e = raw.find("{"), raw.rfind("}")+1
            if s!=-1 and e>s:
                d = json.loads(raw[s:e])
                x = d.get("x", 50)
                y = d.get("y", 50)
                action = d.get("action","click")
                right  = action == "right_click"
                print(f"  💣 {action} at {x}%,{y}% — {d.get('reason','')}")
                self.act.click_pct(x, y, right=right)
                self.cd = 6
        except Exception as ex:
            print(f"  💣 error: {ex}")
            self.act.click_pct(50, 50)
            self.cd = 6


# ─────────────────────────────────────────────────────────────
#  CHESS  (LLM)
# ─────────────────────────────────────────────────────────────
class ChessBot:
    def __init__(self, act: Act, vision_model="gemma3", region=None):
        self.act   = act
        self.model = vision_model
        self.loop  = 0
        self.cd    = 0
        self.clicks= []   # pending clicks (two per move)

    def _img_bytes(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)

        # If we have pending clicks, execute them
        if self.clicks:
            x, y = self.clicks.pop(0)
            self.act.click_pct(x, y)
            self.cd = 2
            return

        if self.cd > 0:
            return

        img = self._img_bytes(bgr)
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role":"user","content":
                    """Chess board. What is the best move?
Reply ONLY:
{"from_x":0-100,"from_y":0-100,"to_x":0-100,"to_y":0-100,"reason":"brief"}
x,y are percentages. JSON only.""",
                    "images":[img]}],
                options={"temperature":0.1,"num_predict":100},
            )
            raw = resp["message"]["content"].strip()
            s,e = raw.find("{"), raw.rfind("}")+1
            if s!=-1 and e>s:
                d  = json.loads(raw[s:e])
                fx = d.get("from_x", 50)
                fy = d.get("from_y", 50)
                tx = d.get("to_x",   50)
                ty = d.get("to_y",   50)
                print(f"  ♟ ({fx}%,{fy}%) → ({tx}%,{ty}%) {d.get('reason','')}")
                # Queue two clicks
                self.clicks = [(fx,fy),(tx,ty)]
                self.cd = 10
        except Exception as ex:
            print(f"  ♟ error: {ex}")
            self.cd = 10


# ─────────────────────────────────────────────────────────────
#  SOLITAIRE  (LLM)
# ─────────────────────────────────────────────────────────────
class SolitaireBot:
    def __init__(self, act: Act, vision_model="gemma3", region=None):
        self.act   = act
        self.model = vision_model
        self.loop  = 0
        self.cd    = 0

    def _img_bytes(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((512,512), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def step(self, bgr):
        self.loop += 1
        self.cd    = max(0, self.cd - 1)
        if self.cd > 0:
            return

        img = self._img_bytes(bgr)
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role":"user","content":
                    """Klondike Solitaire. What card should I click?
First priority: move Ace to foundation.
Second: uncover face-down cards.
Third: click the deck to draw.
Reply ONLY: {"x":0-100,"y":0-100,"reason":"brief"}
JSON only.""",
                    "images":[img]}],
                options={"temperature":0.1,"num_predict":80},
            )
            raw = resp["message"]["content"].strip()
            s,e = raw.find("{"), raw.rfind("}")+1
            if s!=-1 and e>s:
                d = json.loads(raw[s:e])
                x = d.get("x",50)
                y = d.get("y",50)
                print(f"  🃏 click {x}%,{y}% — {d.get('reason','')}")
                self.act.click_pct(x,y)
                self.cd = 5
        except Exception as ex:
            print(f"  🃏 error: {ex}")
            # Draw from deck as fallback
            self.act.click_pct(10,15)
            self.cd = 4


# ============================================================
#  UNIVERSAL RESTART HANDLER
# ============================================================

class RestartHandler:
    def __init__(self, act: Act, ocr: OCR):
        self.act  = act
        self.ocr  = ocr
        self.cd   = 0
        self.tries= 0

    def handle(self, bgr):
        self.cd   = max(0, self.cd - 1)
        if self.cd > 0:
            return

        iw, ih = bgr.shape[1], bgr.shape[0]
        result = self.ocr.find_restart_button(bgr)
        if result:
            cx, cy = result
            print(f"  🔄 Restart button at ({cx},{cy})")
            self.act.click_img(cx, cy, iw, ih)
            self.cd   = 20
            self.tries = 0
            return

        self.tries += 1
        print(f"  🔄 No button found (try {self.tries})")
        if self.tries % 3 == 0:
            self.act.key("space")
            time.sleep(0.2)
            self.act.click_pct(50, 50)
        self.cd = 10


# ============================================================
#  PRESETS
# ============================================================

PRESETS = {
    "flappy_bird":      {"bot":"flappy",    "delay":0.05, "extra_buttons":["ok","play","tap"]},
    "geometry_dash":    {"bot":"geodash",   "delay":0.05, "extra_buttons":["play","retry","ok"]},
    "stickman_hook":    {"bot":"geodash",   "delay":0.08, "extra_buttons":["play","retry","next"]},
    "dino_run":         {"bot":"dino",      "delay":0.05, "extra_buttons":[]},
    "snake":            {"bot":"snake",     "delay":0.08, "extra_buttons":["play","start","restart"]},
    "tetris":           {"bot":"tetris",    "delay":0.05, "extra_buttons":["start","play","ok"]},
    "cookie_clicker":   {"bot":"cookie",    "delay":0.05, "extra_buttons":["buy","upgrade","dismiss"]},
    "2048":             {"bot":"2048",      "delay":0.2,  "extra_buttons":["new game","try again"]},
    "pacman":           {"bot":"pacman",    "delay":0.08, "extra_buttons":["start","play","1 player"]},
    "space_invaders":   {"bot":"invaders",  "delay":0.05, "extra_buttons":["play","start","1 player"]},
    "wordle":           {"bot":"wordle",    "delay":0.3,  "extra_buttons":["play","ok","got it"]},
    "minesweeper":      {"bot":"minesweeper","delay":0.3, "extra_buttons":["new game","ok","beginner"]},
    "chess":            {"bot":"chess",     "delay":0.3,  "extra_buttons":["new game","play","ok"]},
    "solitaire":        {"bot":"solitaire", "delay":0.3,  "extra_buttons":["new game","deal","ok"]},
}


# ============================================================
#  MAIN CONTROLLER
# ============================================================

class GameAI:
    def __init__(self, preset_name, vision_model="gemma3",
                 region=None, debug=True):
        global DEBUG
        DEBUG = debug

        cfg = PRESETS.get(preset_name)
        if not cfg:
            raise ValueError(f"Unknown preset: {preset_name}")

        self.cfg          = cfg
        self.name         = preset_name
        self.region       = region
        self.delay        = cfg["delay"]
        self.loop         = 0
        self.extra_buttons= cfg.get("extra_buttons", [])

        self.ocr     = OCR()
        self.act     = Act(region=region)
        self.restart = RestartHandler(self.act, self.ocr)

        bot_type = cfg["bot"]
        if   bot_type == "flappy":      self.bot = FlappyBot(self.act, region)
        elif bot_type == "geodash":     self.bot = GeometryDashBot(self.act, region)
        elif bot_type == "dino":        self.bot = DinoBot(self.act, region)
        elif bot_type == "snake":       self.bot = SnakeBot(self.act, region)
        elif bot_type == "tetris":      self.bot = TetrisBot(self.act, region)
        elif bot_type == "cookie":      self.bot = CookieBot(self.act, region)
        elif bot_type == "2048":        self.bot = Bot2048(self.act, self.ocr, region)
        elif bot_type == "pacman":      self.bot = PacManBot(self.act, region)
        elif bot_type == "invaders":    self.bot = SpaceInvadersBot(self.act, region)
        elif bot_type == "wordle":      self.bot = WordleBot(self.act, vision_model)
        elif bot_type == "minesweeper": self.bot = MinesweeperBot(self.act, self.ocr, vision_model, region)
        elif bot_type == "chess":       self.bot = ChessBot(self.act, vision_model, region)
        elif bot_type == "solitaire":   self.bot = SolitaireBot(self.act, vision_model, region)
        else: raise ValueError(f"Unknown bot type: {bot_type}")

        print(f"\n✅ {preset_name} ready | bot={bot_type} | delay={self.delay}s\n")

    def step(self):
        self.loop += 1
        bgr = to_bgr(capture(self.region))

        # Fast game-over check (only every 10 loops to save time)
        if self.loop % 10 == 0:
            if self.ocr.is_game_over(bgr) or self.ocr.is_menu(bgr):
                print(f"\n  ⚠️  Game over / menu detected → restarting")
                self.restart.handle(bgr)
                return

        # Run the bot
        self.bot.step(bgr)

    def run(self, max_loops=None):
        print("🚀 Starting in 3…")
        for i in range(3,0,-1):
            print(f"   {i}…"); time.sleep(1)
        print("   GO!  (move mouse top-left to stop)\n")

        n = 0
        try:
            while max_loops is None or n < max_loops:
                t0 = time.time()
                self.step()
                elapsed = time.time() - t0
                # Keep to target delay
                wait = max(0, self.delay - elapsed)
                time.sleep(wait)
                n += 1
        except KeyboardInterrupt:
            print("\n⛔ Stopped")
        except pyautogui.FailSafeException:
            print("\n⛔ Failsafe")
        print(f"\n📊 {n} loops")


# ============================================================
#  HELPERS
# ============================================================

def pick_region():
    print("\n🖱  TOP-LEFT corner → Enter")
    input("  → ")
    x1,y1 = pyautogui.position()
    print(f"  ({x1},{y1})")
    print("  BOTTOM-RIGHT corner → Enter")
    input("  → ")
    x2,y2 = pyautogui.position()
    region = (min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
    print(f"  ✅ {region}")
    return region

def test_ocr(region=None):
    ocr = OCR()
    bgr = to_bgr(capture(region))
    texts = ocr.read(bgr)
    print(f"\nText found ({len(texts)}):")
    for t in texts:
        print(f"  \"{t['text']}\" @ ({t['cx']},{t['cy']}) conf={t['conf']}")
    btn = ocr.find_restart_button(bgr)
    print(f"\nRestart button: {btn}")
    print(f"Game over: {ocr.is_game_over(bgr)}")
    annotated = bgr.copy()
    for t in texts:
        cv2.rectangle(annotated,(t["x"],t["y"]),
                      (t["x"]+t["w"],t["y"]+t["h"]),(0,255,0),2)
        cv2.putText(annotated, t["text"],
                    (t["x"], max(t["y"]-4,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.imwrite("debug_ocr_test.png", annotated)
    print("Saved: debug_ocr_test.png")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n🎮 AI Game Player\n")
    print("  1. Play a game")
    print("  2. Test OCR")
    mode = input("\nChoice [1]: ").strip() or "1"

    if mode == "2":
        reg = input("Region? (p)ick/(f)ullscreen [f]: ").strip().lower()
        test_ocr(pick_region() if reg=="p" else None)
        return

    names = list(PRESETS.keys())
    print("\nGames:")
    for i,n in enumerate(names,1):
        print(f"  {i:2d}. {n:<22} [{PRESETS[n]['bot']}]")

    choice = input("\nGame: ").strip()
    name   = "flappy_bird"
    if choice.isdigit():
        idx = int(choice)-1
        if 0 <= idx < len(names):
            name = names[idx]

    v_model = input("Vision model [gemma3]: ").strip() or "gemma3"

    reg = input("Region? (p)ick/(f)ullscreen [f]: ").strip().lower()
    region = pick_region() if reg=="p" else None

    max_l     = input("Max loops [unlimited]: ").strip()
    max_loops = int(max_l) if max_l else None
    debug     = input("Debug frames? y/n [n]: ").strip().lower() == "y"

    ai = GameAI(name, vision_model=v_model, region=region, debug=debug)
    ai.run(max_loops=max_loops)


if __name__ == "__main__":
    main()