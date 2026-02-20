from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List

from app.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_SEC, logger

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _is_enabled() -> bool:
    return bool(OPENAI_API_KEY)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    # Handle model outputs that wrap JSON with markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _build_prompt(
    user_profile: Dict[str, float],
    recommendations: List[Dict[str, Any]],
) -> str:
    compact_items = []
    for item in recommendations:
        compact_items.append(
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "artist": item.get("artist"),
                "match_percent": item.get("match_percent"),
                "difficulty": item.get("difficulty"),
                "range_level": item.get("range_level"),
                "tags": item.get("tags", []),
            }
        )

    return (
        "너는 보컬 코치다. 추천곡별로 사용자 맞춤 추천 이유를 한국어로 2개씩 생성해라.\\n"
        "출력은 반드시 JSON 객체 하나만 반환한다. 다른 텍스트를 추가하지 마라.\\n"
        '형식: {"reasons_by_rank": {"1": ["...", "..."], "2": ["...", "..."]}}\\n'
        "규칙: 각 문장은 25자 내외, 과장 금지, 실용적인 발성 관점 포함.\\n"
        f"user_profile={json.dumps(user_profile, ensure_ascii=False)}\\n"
        f"recommendations={json.dumps(compact_items, ensure_ascii=False)}"
    )


def enrich_recommendation_reasons(
    user_profile: Dict[str, float],
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not recommendations:
        logger.info("openai reason enrichment skipped: empty recommendations")
        return recommendations

    if not _is_enabled():
        logger.info("openai reason enrichment skipped: OPENAI_API_KEY is not set")
        return recommendations

    try:
        started = time.perf_counter()
        logger.info(
            "openai reason enrichment start model=%s items=%d",
            OPENAI_MODEL,
            len(recommendations),
        )
        prompt = _build_prompt(user_profile, recommendations)
        body = {
            "model": OPENAI_MODEL,
            "temperature": 0.4,
            "messages": [
                {
                    "role": "system",
                    "content": "정확한 JSON만 출력한다.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 350,
        }

        req = urllib.request.Request(
            OPENAI_API_URL,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT_SEC) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        content = ""
        choices = raw.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        parsed = _extract_json(content)
        if not parsed:
            logger.warning("openai reason enrichment skipped: invalid json response")
            return recommendations

        reasons_by_rank = parsed.get("reasons_by_rank", {})
        if not isinstance(reasons_by_rank, dict):
            logger.warning("openai reason enrichment skipped: reasons_by_rank is not dict")
            return recommendations

        updated_count = 0
        for item in recommendations:
            rank = str(item.get("rank"))
            reasons = reasons_by_rank.get(rank)
            if isinstance(reasons, list):
                cleaned = [str(x).strip() for x in reasons if str(x).strip()]
                if cleaned:
                    item["reasons"] = cleaned[:2]
                    updated_count += 1

        logger.info(
            "openai reason enrichment done updated=%d/%d took=%.3fs",
            updated_count,
            len(recommendations),
            time.perf_counter() - started,
        )

        return recommendations
    except Exception as err:
        logger.warning("openai reason enrichment failed err=%s", err)
        return recommendations
