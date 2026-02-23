"""Global configuration."""

# Ollama settings
OLLAMA_MODEL = "llava"  # Vision-capable model (CRITICAL - needs to "see")
OLLAMA_HOST = "http://localhost:11434"

# Screen capture settings
DEFAULT_MONITOR = 1  # Primary monitor
CAPTURE_FPS = 2      # Screenshots per second (LLM is slow, start low)

# Input settings
INPUT_DELAY = 0.05   # Delay between actions
PYDIRECTINPUT_PAUSE = 0.0

# Agent settings
MAX_MEMORY_FRAMES = 10  # How many past states to remember
GAME_LOOP_DELAY = 0.5   # Seconds between AI decisions