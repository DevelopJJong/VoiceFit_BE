from __future__ import annotations

import logging
import os
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
SILENT_AUDIO_RMS_THRESHOLD = 0.0015
CLIPPING_SAMPLE_THRESHOLD = 0.01
CLIPPING_VALUE = 0.99

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_TIMEOUT_SEC = float(os.getenv("SPOTIFY_TIMEOUT_SEC", "5"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "8"))
OPENAI_ENRICH_ENABLED = os.getenv("OPENAI_ENRICH_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
OPENAI_MAX_RETRIES = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
OPENAI_BACKOFF_BASE_SEC = max(0.1, float(os.getenv("OPENAI_BACKOFF_BASE_SEC", "1.0")))
OPENAI_CACHE_TTL_SEC = max(0, int(os.getenv("OPENAI_CACHE_TTL_SEC", "300")))

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
