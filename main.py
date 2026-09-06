"""LunarAlign — Launch the web dashboard."""
import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 60)
    print("  LunarAlign — Multi-Modal Lunar Image Registration")
    print("  Smart India Hackathon 2026 • SIH26166")
    print("=" * 60)
    print("\n  Starting dashboard at http://localhost:8000\n")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=False)
