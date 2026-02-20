from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BACKOFF_BASE_SEC,
    OPENAI_CACHE_TTL_SEC,
    OPENAI_ENRICH_ENABLED,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SEC,
    logger,
)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _is_enabled() -> bool:
    return OPENAI_ENRICH_ENABLED and bool(OPENAI_API_KEY)


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


def _cache_key(user_profile: Dict[str, float], recommendations: List[Dict[str, Any]]) -> str:
    payload = {"user_profile": user_profile, "recommendations": recommendations}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(key: str) -> dict[str, Any] | None:
    if OPENAI_CACHE_TTL_SEC <= 0:
        return None
    cached = _CACHE.get(key)
    if not cached:
        return None
    saved_at, value = cached
    age = time.time() - saved_at
    if age > OPENAI_CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return value


def _write_cache(key: str, value: dict[str, Any]) -> None:
    if OPENAI_CACHE_TTL_SEC <= 0:
        return
    _CACHE[key] = (time.time(), value)


def _request_openai(body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    last_err: Exception | None = None
    attempts = OPENAI_MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT_SEC) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code != 429 or attempt >= attempts:
                raise
            retry_after = err.headers.get("Retry-After", "").strip()
            sleep_sec = OPENAI_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            if retry_after:
                try:
                    sleep_sec = max(sleep_sec, float(retry_after))
                except ValueError:
                    pass
            logger.warning(
                "openai reason enrichment rate-limited status=429 attempt=%d/%d sleep=%.2fs",
                attempt,
                attempts,
                sleep_sec,
            )
            time.sleep(sleep_sec)
        except Exception as err:
            last_err = err
            if attempt >= attempts:
                raise
            sleep_sec = OPENAI_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "openai reason enrichment retry after error attempt=%d/%d sleep=%.2fs err=%s",
                attempt,
                attempts,
                sleep_sec,
                err,
            )
            time.sleep(sleep_sec)
    if last_err:
        raise last_err
    raise RuntimeError("openai request failed without error")


def enrich_recommendation_reasons(
    user_profile: Dict[str, float],
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not recommendations:
        logger.info("openai reason enrichment skipped: empty recommendations")
        return recommendations

    if not _is_enabled():
        logger.info("openai reason enrichment skipped: disabled or OPENAI_API_KEY is not set")
        return recommendations

    try:
        started = time.perf_counter()
        cache_key = _cache_key(user_profile, recommendations)
        cached_payload = _read_cache(cache_key)
        if cached_payload:
            logger.info("openai reason enrichment cache_hit ttl=%ds", OPENAI_CACHE_TTL_SEC)
            reasons_by_rank = cached_payload.get("reasons_by_rank", {})
            if isinstance(reasons_by_rank, dict):
                for item in recommendations:
                    rank = str(item.get("rank"))
                    reasons = reasons_by_rank.get(rank)
                    if isinstance(reasons, list):
                        cleaned = [str(x).strip() for x in reasons if str(x).strip()]
                        if cleaned:
                            item["reasons"] = cleaned[:2]
            return recommendations

        logger.info(
            "openai reason enrichment start model=%s items=%d retries=%d",
            OPENAI_MODEL,
            len(recommendations),
            OPENAI_MAX_RETRIES,
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
        raw = _request_openai(body)

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

        _write_cache(cache_key, {"reasons_by_rank": reasons_by_rank})
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
