from __future__ import annotations

from typing import Any, Dict, List


def _similarity(user: Dict[str, float], song: Dict[str, Any]) -> float:
    ub, uh, us = user["brightness"], user["husky"], user["softness"]
    sb, sh, ss = song["brightness"], song["husky"], song["softness"]
    sim = 0.4 * (1 - abs(ub - sb)) + 0.3 * (1 - abs(uh - sh)) + 0.3 * (1 - abs(us - ss))
    return max(0.0, min(1.0, float(sim)))


def _is_range_allowed(
    song_target: str,
    vocal_range_mode: str,
    allow_cross_gender: bool,
) -> bool:
    if vocal_range_mode == "any":
        return True
    if allow_cross_gender:
        return True
    return song_target == vocal_range_mode or song_target == "any"


def _make_reasons(user: Dict[str, float], song: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []

    b_diff = abs(user["brightness"] - song["brightness"])
    if b_diff <= 0.22:
        if user["brightness"] >= 0.5:
            reasons.append("밝은 톤과 잘 어울림")
        else:
            reasons.append("따뜻한 톤과 잘 어울림")

    if user["husky"] < 0.5:
        reasons.append("부드러운 발성에 적합")
    else:
        reasons.append("거친 톤이 잘 살아남")

    if user["softness"] >= 0.55:
        reasons.append("잔잔한 곡에서 톤이 잘 살아남")
    else:
        reasons.append("에너지 있는 구간에서 존재감이 좋음")

    return reasons[:2]


def recommend_songs(
    user_profile: Dict[str, float],
    songs: List[Dict[str, Any]],
    vocal_range_mode: str,
    allow_cross_gender: bool,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for song in songs:
        if not _is_range_allowed(song.get("target_range", "any"), vocal_range_mode, allow_cross_gender):
            continue

        sim = _similarity(user_profile, song)
        candidates.append(
            {
                "title": song["title"],
                "artist": song["artist"],
                "score": round(sim, 4),
                "match_percent": round(sim * 100),
                "reasons": _make_reasons(user_profile, song),
                "tags": song.get("tags", []),
                "difficulty": song.get("difficulty", 2),
                "range_level": song.get("range_level", 2),
                "external_url": song.get("external_url"),
                "cover_url": song.get("cover_url"),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    top = candidates[:top_k]
    for i, item in enumerate(top, start=1):
        item["rank"] = i

    return top
