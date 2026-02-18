from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SONG_DB_PATH = DATA_DIR / "song_db.json"

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm", "m4a"}
TARGET_SR = 16000
MAX_DURATION_SEC = 20.0
MIN_DURATION_SEC = 3.0
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

RMS_TRIM_THRESHOLD = 0.01
BAD_SIGNAL_RMS_THRESHOLD = 0.005
CLIPPING_SAMPLE_THRESHOLD = 0.01
CLIPPING_VALUE = 0.99

CORS_ORIGINS = [
    "http://localhost:5173",
    "https://voice-fit-fe.vercel.app",
]


@dataclass
class AppError(Exception):
    code: str
    message: str
    hint: str
    status_code: int = 400


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voicefit")
