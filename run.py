"""Dev server entry point — works on Windows without PYTHONPATH tricks.

Usage:
    python run.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work correctly
# on Windows where the editable-install path may not be picked up by uvicorn's
# reload subprocess.
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # set True only if running on Linux/macOS
    )
