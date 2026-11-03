"""Vercel Python Serverless Function entrypoint.

Routes all `/api/*` requests through `MotionGuidanceHandler` in `server.py`.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import MotionGuidanceHandler  # noqa: E402


class handler(MotionGuidanceHandler):
    pass

