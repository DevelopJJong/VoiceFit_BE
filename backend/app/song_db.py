from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

from app.config import SONG_DB_PATH, AppError, logger


@lru_cache(maxsize=1)
def load_song_db() -> List[Dict[str, Any]]:
    if not SONG_DB_PATH.exists():
        raise AppError(
            code="SONG_DB_NOT_FOUND",
            message="song_db.json 파일을 찾을 수 없습니다.",
            hint="backend/data/song_db.json 파일 경로를 확인하세요.",
            status_code=500,
        )

    try:
        with SONG_DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        logger.exception("song db json decode error")
        raise AppError(
            code="SONG_DB_INVALID",
            message="song_db.json 형식이 올바르지 않습니다.",
            hint=f"JSON 파싱 오류: {err}",
            status_code=500,
        ) from err

    if not isinstance(data, list):
        raise AppError(
            code="SONG_DB_INVALID",
            message="song_db.json 루트는 배열이어야 합니다.",
            hint="[{...}, {...}] 형태인지 확인하세요.",
            status_code=500,
        )

    if len(data) < 300:
        raise AppError(
            code="SONG_DB_TOO_SMALL",
            message="song_db.json 곡 수가 부족합니다.",
            hint="최소 300곡 이상이 필요합니다.",
            status_code=500,
        )

    male_count = sum(1 for x in data if x.get("target_range") == "male")
    female_count = sum(1 for x in data if x.get("target_range") == "female")
    if male_count == 0 or female_count == 0:
        raise AppError(
            code="SONG_DB_RANGE_INVALID",
            message="남/여 타겟 곡 분리가 필요합니다.",
            hint="target_range가 male/female인 곡을 각각 포함하세요.",
            status_code=500,
        )
    return data
