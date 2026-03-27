import ollama
import io
import sys
import time
from PIL import ImageGrab, ImageEnhance, Image


def capture_screen(region=None, enhance=True):
    """Capture and preprocess screen for optimal model consumption."""
    screenshot = ImageGrab.grab(bbox=region)
    original_size = screenshot.size

    max_dim = 1344
    w, h = screenshot.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        screenshot = screenshot.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    if enhance:
        screenshot = ImageEnhance.Contrast(screenshot).enhance(1.3)
        screenshot = ImageEnhance.Sharpness(screenshot).enhance(1.5)
        screenshot = ImageEnhance.Brightness(screenshot).enhance(1.1)

    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue(), original_size, screenshot.size


SYSTEM_PROMPT = """You are looking at a screen capture. 

Look at what's actually there. Identify the game if you can. Describe what 
matters — what a player would care about right now in this moment.

Don't list things that aren't visible. Don't speculate about things you 
can't see. Don't pad your response with generic observations.

If you see specific numbers, text, or values on screen, read them accurately.

Be concise. Only mention what's relevant."""


def list_available_models():
    """Show what models are installed in Ollama."""
    try:
        models = ollama.list()
        print("\n📋 Installed models:")
        for m in models["models"]:
            print(f"   - {m['name']}")
        print()
    except Exception as e:
        print(f"Could not list models: {e}")


def ask_about_screen(prompt=None,
                     model="llava",
                     region=None,
                     stream=True,
                     temperature=0.1):
    """Capture screen and ask the AI about it."""
    print("📸 Capturing screen...")
    image_bytes, orig_size, processed_size = capture_screen(
        region=region, enhance=True
    )
    print(f"   {orig_size[0]}x{orig_size[1]} → {processed_size[0]}x{processed_size[1]}")

    if not prompt:
        user_message = "What's happening here?"
    else:
        user_message = prompt

    print(f"🤖 Asking {model}...\n")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
            "images": [image_bytes],
        },
    ]

    options = {
        "temperature": temperature,
        "num_predict": 1024,
    }

    try:
        if stream:
            response_stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options,
            )

            full_response = ""
            for chunk in response_stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_response += token

            print()
            return full_response
        else:
            response = ollama.chat(
                model=model,
                messages=messages,
                options=options,
            )
            result = response["message"]["content"]
            print(result)
            return result

    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg or "404" in error_msg:
            print(f"\n❌ Model '{model}' not found!")
            print(f"\nTry installing it:")
            print(f"   ollama pull {model}")
            list_available_models()
        else:
            print(f"\n❌ Error: {e}")
        return ""


def compare_frames(model="llava", delay=2.0):
    """Capture two frames to understand what's changing."""
    print("📸 Frame 1...")
    frame1, _, _ = capture_screen(enhance=True)

    print(f"⏳ Waiting {delay}s...")
    time.sleep(delay)

    print("📸 Frame 2...")
    frame2, _, _ = capture_screen(enhance=True)

    print(f"🤖 Analyzing with {model}...\n")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "These are two frames a few seconds apart. What changed?",
                    "images": [frame1, frame2],
                },
            ],
            options={"temperature": 0.1},
        )

        result = response["message"]["content"]
        print(result)
        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return ""


def interactive_mode(model="llava"):
    """Continuously ask questions about your screen."""
    print("=" * 40)
    print("  🎮 Game Screen Vision")
    print(f"  Model: {model}")
    print()
    print("  Just type a question, or:")
    print("    Enter     — describe what's on screen")
    print("    compare   — capture two frames")
    print("    models    — list installed models")
    print("    quit      — exit")
    print("=" * 40)

    while True:
        try:
            user_input = input("\n🎯 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "models":
            list_available_models()
            continue

        if user_input.lower() == "compare":
            print("-" * 40)
            compare_frames(model=model)
            print("-" * 40)
            continue

        print("-" * 40)
        ask_about_screen(
            prompt=user_input if user_input else None,
            model=model,
        )
        print("-" * 40)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "llava"
    interactive_mode(model=model)