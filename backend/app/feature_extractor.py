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


def _estimate_pitch_stats(frames: np.ndarray, sr: int) -> tuple[float, float, float]:
    min_f0 = 70.0
    max_f0 = 450.0
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)

    f0s: list[float] = []
    voiced_scores: list[float] = []
    for frame in frames:
        x = frame - np.mean(frame)
        energy = np.sum(np.square(x))
        if energy < 1e-6:
            continue
        corr = np.correlate(x, x, mode="full")
        corr = corr[corr.size // 2 :]
        if max_lag >= corr.size:
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx] / (corr[0] + 1e-10))
        if peak_val < 0.25:
            continue
        lag = peak_idx + min_lag
        f0s.append(float(sr / lag))
        voiced_scores.append(peak_val)

    if not f0s:
        return 0.0, 0.0, 0.0
    f0 = np.asarray(f0s, dtype=np.float32)
    voiced = np.asarray(voiced_scores, dtype=np.float32)
    return float(np.mean(f0)), float(np.std(f0)), float(np.mean(voiced))


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
    flatness = np.exp(np.mean(np.log(power), axis=1)) / (np.mean(power, axis=1) + 1e-10)
    pitch_mean, pitch_std, voiced_score = _estimate_pitch_stats(frames, sr)

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
        "flatness_mean": float(np.mean(flatness)),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "voiced_score": voiced_score,
    }


def profile_from_features(features: Dict[str, float]) -> Dict[str, float]:
    centroid = features["spectral_centroid_mean"]
    rolloff = features["spectral_rolloff_mean"]
    zcr = features["zcr_mean"]
    rms = features["rms_mean"]
    flatness = features.get("flatness_mean", 0.0)
    pitch_mean = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    voiced = features.get("voiced_score", 0.0)

    centroid_n = _normalize(centroid, 500.0, 3500.0)
    rolloff_n = _normalize(rolloff, 1200.0, 6500.0)
    pitch_n = _normalize(pitch_mean, 90.0, 320.0)
    zcr_n = _normalize(zcr, 0.02, 0.20)
    rms_n = _normalize(rms, 0.01, 0.20)
    flat_n = _normalize(flatness, 0.03, 0.45)
    pitch_std_n = _normalize(pitch_std, 5.0, 70.0)
    voiced_n = _normalize(voiced, 0.20, 0.80)

    brightness = 0.45 * centroid_n + 0.35 * rolloff_n + 0.20 * pitch_n
    husky = 0.55 * zcr_n + 0.30 * flat_n + 0.15 * (1.0 - voiced_n)
    softness = 0.50 * (1.0 - rms_n) + 0.25 * (1.0 - zcr_n) + 0.15 * (1.0 - flat_n) + 0.10 * (1.0 - pitch_std_n)

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
    voiced_score: float = 0.0,
    pitch_std: float = 0.0,
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
    if voiced_score > 0:
        score += _normalize(voiced_score, 0.2, 0.8) * 0.08
    if pitch_std > 0:
        score -= _normalize(pitch_std, 40.0, 120.0) * 0.05

    return _clamp01(score)
