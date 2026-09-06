"""LunarAlign — Launch the web dashboard (React + FastAPI)."""
from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


def open_browser() -> None:
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    print("=" * 60)
    print("  LunarAlign — Multi-Modal Lunar Image Registration")
    print("  Smart India Hackathon 2026 • SIH26166 • Team Six Seven")
    print("=" * 60)
    print()

    if FRONTEND_DIST.exists():
        print("  React frontend (production build) detected.")
        print("  Serving dashboard at http://localhost:8000")
    else:
        print("  React frontend has NOT been built yet.")
        print("  To build it, run:")
        print()
        print("    cd frontend")
        print("    npm install")
        print("    npm run build")
        print()
        print("  For live development with hot-reload, run in a separate terminal:")
        print()
        print("    cd frontend && npm run dev")
        print()
        print("  Starting FastAPI backend at http://localhost:8000 ...")

    print()
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=False)
