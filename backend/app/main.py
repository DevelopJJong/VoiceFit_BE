from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.audio_utils import load_audio_from_upload
from app.config import CORS_ORIGINS, AppError, logger
from app.feature_extractor import (
    compute_confidence,
    extract_features,
    profile_from_features,
    summarize_profile,
)
from app.recommender import recommend_songs
from app.schemas import AnalyzeResponse, ErrorResponse
from app.song_db import load_song_db

app = FastAPI(title="VoiceFit Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_headers(request: Request, call_next: Any):
    origin = request.headers.get("origin", "")
    if request.method == "OPTIONS":
        if origin in CORS_ORIGINS:
            response = JSONResponse(status_code=200, content={"ok": True})
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Vary"] = "Origin"
            return response
    response = await call_next(request)
    if origin in CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Vary"] = "Origin"
    return response


def _error_response(status_code: int, code: str, message: str, hint: str) -> JSONResponse:
    payload = ErrorResponse(error={"code": code, "message": message, "hint": hint}).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def _parse_bool(value: str, default: bool = False) -> bool:
    normalized = (value or "").strip().lower()
    if normalized == "":
        return default
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise AppError(
        code="INVALID_BOOLEAN",
        message="불리언 파라미터 형식이 잘못되었습니다.",
        hint="true 또는 false 문자열을 사용하세요.",
        status_code=400,
    )


def _build_mock_response(vocal_range_mode: str, allow_cross_gender: bool) -> dict[str, Any]:
    return {
        "profile": {"brightness": 0.72, "husky": 0.28, "softness": 0.61},
        "summary": "맑고 비교적 밝은 톤이며, 발성이 부드러운 편입니다.",
        "confidence": 0.84,
        "input_info": {
            "duration_sec": 12.4,
            "signal_quality": "good",
            "note": "mock 모드 결과입니다.",
        },
        "filters": {
            "vocal_range_mode": vocal_range_mode,
            "allow_cross_gender": allow_cross_gender,
        },
        "recommendations": [
            {
                "rank": 1,
                "title": "Song A",
                "artist": "Artist A",
                "score": 0.89,
                "match_percent": 89,
                "reasons": ["밝은 톤과 잘 어울림", "부드러운 발성에 적합"],
                "tags": ["bright", "soft"],
                "difficulty": 2,
                "range_level": 2,
                "external_url": "https://example.com/song-a",
                "cover_url": "https://picsum.photos/seed/songa/400/400",
            }
        ],
    }


@app.exception_handler(AppError)
async def app_error_handler(_: Any, exc: AppError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.hint)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Any, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        422,
        "VALIDATION_ERROR",
        "요청 형식이 올바르지 않습니다.",
        str(exc.errors()),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Any, exc: HTTPException) -> JSONResponse:
    return _error_response(
        exc.status_code,
        "HTTP_ERROR",
        str(exc.detail),
        "요청 파라미터 또는 API 경로를 확인하세요.",
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Any, exc: Exception) -> JSONResponse:
    logger.exception("unexpected server error", exc_info=exc)
    return _error_response(
        500,
        "INTERNAL_ERROR",
        "서버 내부 오류가 발생했습니다.",
        "서버 로그를 확인하고 다시 시도하세요.",
    )


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    vocal_range_mode: str = Form("any"),
    allow_cross_gender: str = Form("false"),
    mock: str = Form("false"),
) -> dict[str, Any]:
    vocal_range_mode = vocal_range_mode.strip().lower()
    if vocal_range_mode not in {"male", "female", "any"}:
        raise AppError(
            code="INVALID_VOCAL_RANGE_MODE",
            message="vocal_range_mode 값이 잘못되었습니다.",
            hint='"male", "female", "any" 중 하나를 사용하세요.',
            status_code=400,
        )

    allow_cross_gender_bool = _parse_bool(allow_cross_gender, default=False)
    is_mock = _parse_bool(mock, default=False)

    if is_mock:
        return _build_mock_response(vocal_range_mode, allow_cross_gender_bool)

    audio_result = load_audio_from_upload(file)
    features = extract_features(audio_result.waveform, audio_result.sr)
    profile = profile_from_features(features)
    summary = summarize_profile(profile)
    confidence = compute_confidence(
        duration_sec=audio_result.duration_sec,
        signal_quality=audio_result.signal_quality,
        rms_mean=audio_result.rms_mean,
        clipping_ratio=audio_result.clipping_ratio,
    )

    songs = load_song_db()
    recommendations = recommend_songs(
        user_profile=profile,
        songs=songs,
        vocal_range_mode=vocal_range_mode,
        allow_cross_gender=allow_cross_gender_bool,
        top_k=5,
    )

    return {
        "profile": profile,
        "summary": summary,
        "confidence": confidence,
        "input_info": {
            "duration_sec": round(audio_result.duration_sec, 2),
            "signal_quality": audio_result.signal_quality,
            "note": audio_result.note,
        },
        "filters": {
            "vocal_range_mode": vocal_range_mode,
            "allow_cross_gender": allow_cross_gender_bool,
        },
        "recommendations": recommendations,
    }
