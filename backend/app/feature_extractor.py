from __future__ import annotations

from typing import Dict

import numpy as np


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return _clamp01((value - min_val) / (max_val - min_val))


def _frame_signal(y: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if y.size < frame_size:
        y = np.pad(y, (0, frame_size - y.size))
    num_frames = 1 + (y.size - frame_size) // hop_size
    idx = np.arange(frame_size)[None, :] + hop_size * np.arange(num_frames)[:, None]
    return y[idx]


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    frame_size = 1024
    hop_size = 256
    frames = _frame_signal(y.astype(np.float32), frame_size, hop_size)
    win = np.hanning(frame_size).astype(np.float32)
    windowed = frames * win[None, :]

    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = np.square(spectrum) + 1e-10
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr).astype(np.float32)

    centroid = np.sum(spectrum * freqs[None, :], axis=1) / (np.sum(spectrum, axis=1) + 1e-10)

    cumsum = np.cumsum(power, axis=1)
    cutoff = 0.85 * cumsum[:, -1][:, None]
    rolloff_idx = np.argmax(cumsum >= cutoff, axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, freqs.size - 1)]

    sign = np.sign(frames)
    zc = np.mean(np.abs(np.diff(sign, axis=1)) > 0, axis=1)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    # Lightweight cepstral proxy for MVP stability (keeps mfcc_mean field contract).
    mean_power = np.mean(power, axis=0)
    cepstrum = np.fft.irfft(np.log(mean_power), n=frame_size)
    mfcc_like = cepstrum[:13]

    return {
        "mfcc_mean": float(np.mean(mfcc_like)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "zcr_mean": float(np.mean(zc)),
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
