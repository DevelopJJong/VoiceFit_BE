from __future__ import annotations

from typing import Dict

import librosa
import numpy as np


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return _clamp01((value - min_val) / (max_val - min_val))


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    return {
        "mfcc_mean": float(np.mean(mfcc)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "rms_mean": float(np.mean(rms)),
    }


def profile_from_features(features: Dict[str, float]) -> Dict[str, float]:
    centroid = features["spectral_centroid_mean"]
    zcr = features["zcr_mean"]
    rms = features["rms_mean"]

    brightness = _normalize(centroid, 500.0, 3500.0)
    husky = _normalize(zcr, 0.02, 0.20)
    softness = 1.0 - _normalize(rms, 0.01, 0.20)

    return {
        "brightness": _clamp01(brightness),
        "husky": _clamp01(husky),
        "softness": _clamp01(softness),
    }


def summarize_profile(profile: Dict[str, float]) -> str:
    brightness = profile["brightness"]
    husky = profile["husky"]
    softness = profile["softness"]

    tone_text = "맑고 비교적 밝은" if brightness >= 0.55 else "따뜻하고 안정적인"
    husky_text = "거친 결이 있는" if husky >= 0.55 else "깨끗한"
    softness_text = "부드러운" if softness >= 0.55 else "직진감 있는"

    return f"{tone_text} 톤이며, {husky_text} 발성 성향과 {softness_text} 표현이 느껴집니다."


def compute_confidence(
    duration_sec: float,
    signal_quality: str,
    rms_mean: float,
    clipping_ratio: float,
) -> float:
    score = 0.45

    score += min(duration_sec / 20.0, 1.0) * 0.30

    if signal_quality == "good":
        score += 0.20
    elif signal_quality == "ok":
        score += 0.10

    if rms_mean < 0.006:
        score -= 0.10
    if clipping_ratio > 0.01:
        score -= 0.10

    return _clamp01(score)
