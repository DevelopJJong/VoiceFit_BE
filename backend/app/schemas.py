from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    hint: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Profile(BaseModel):
    brightness: float = Field(..., ge=0.0, le=1.0)
    husky: float = Field(..., ge=0.0, le=1.0)
    softness: float = Field(..., ge=0.0, le=1.0)


class InputInfo(BaseModel):
    duration_sec: float
    signal_quality: Literal["good", "ok", "bad"]
    note: str


class Filters(BaseModel):
    vocal_range_mode: Literal["male", "female", "any"]
    allow_cross_gender: bool


class RecommendationItem(BaseModel):
    rank: int
    title: str
    artist: str
    score: float
    match_percent: int
    reasons: List[str]
    tags: List[str]
    difficulty: int
    range_level: int
    external_url: Optional[str] = None
    cover_url: Optional[str] = None


class AnalyzeResponse(BaseModel):
    profile: Profile
    summary: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    input_info: InputInfo
    filters: Filters
    recommendations: List[RecommendationItem]


class SongItem(BaseModel):
    title: str
    artist: str
    brightness: float = Field(..., ge=0.0, le=1.0)
    husky: float = Field(..., ge=0.0, le=1.0)
    softness: float = Field(..., ge=0.0, le=1.0)
    range_level: int = Field(..., ge=1, le=3)
    difficulty: int = Field(..., ge=1, le=3)
    target_range: Literal["male", "female", "any"]
    tags: List[str]
    external_url: Optional[str] = None
    cover_url: Optional[str] = None
