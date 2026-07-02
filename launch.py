"""
Launch Script — AI Trading Analyst
=====================================
Run this script to start the application.
Opens the browser automatically at http://127.0.0.1:8765
"""

import sys
import os
import time
import webbrowser
import subprocess
import threading
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check_python_version():
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current:", sys.version)
        sys.exit(1)


def check_dependencies():
    try:
        import fastapi, uvicorn, pandas, yfinance, sqlalchemy
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def open_browser_delayed(url: str, delay: float = 2.5):
    """Open browser after a short delay to let the server start."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def run():
    print("=" * 60)
    print("  [*] AI Trading Analyst -- Indian Markets")
    print("  Version: 1.0.0 | Mode: Simulation Only [SAFE]")
    print("  Running Cost: Rs.0 | No Cloud | Fully Offline")
    print("=" * 60)

    check_python_version()
    check_dependencies()

    from app.config import settings

    url = f"http://{settings.HOST}:{settings.PORT}"
    print(f"\n[START] Starting server at {url}")
    print("[OPEN] Opening browser...")
    open_browser_delayed(url)

    print("\n[GUIDE] Quick Guide:")
    print("  1. Click 'Scan Market' to analyze stocks")
    print("  2. Install Ollama + pull mistral for AI Chat")
    print("  3. All trades are SIMULATION only by default")
    print("\nPress Ctrl+C to stop.\n")

    import uvicorn
    from app.main import app
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    run()
