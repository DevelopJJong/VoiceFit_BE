from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Optional

from app.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_TIMEOUT_SEC, logger

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"


def _is_enabled() -> bool:
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)


@lru_cache(maxsize=1)
def _get_access_token() -> str:
    if not _is_enabled():
        return ""

    raw = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode("utf-8")
    basic_auth = base64.b64encode(raw).decode("ascii")

    req = urllib.request.Request(
        SPOTIFY_TOKEN_URL,
        data=b"grant_type=client_credentials",
        method="POST",
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    with urllib.request.urlopen(req, timeout=SPOTIFY_TIMEOUT_SEC) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("access_token", "")


def find_cover_url(title: str, artist: str) -> Optional[str]:
    if not _is_enabled():
        return None

    try:
        token = _get_access_token()
        if not token:
            return None

        query = f"track:{title} artist:{artist}"
        qs = urllib.parse.urlencode(
            {
                "q": query,
                "type": "track",
                "limit": 1,
            }
        )
        req = urllib.request.Request(
            f"{SPOTIFY_SEARCH_URL}?{qs}",
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=SPOTIFY_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        items = payload.get("tracks", {}).get("items", [])
        if not items:
            return None

        images = items[0].get("album", {}).get("images", [])
        if not images:
            return None
        return images[0].get("url")
    except Exception as err:
        logger.warning("spotify cover lookup failed title=%s artist=%s err=%s", title, artist, err)
        return None
