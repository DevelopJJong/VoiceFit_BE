from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import UploadFile

from app.config import (
    ALLOWED_EXTENSIONS,
    BAD_SIGNAL_RMS_THRESHOLD,
    CLIPPING_SAMPLE_THRESHOLD,
    CLIPPING_VALUE,
    MAX_UPLOAD_BYTES,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    RMS_TRIM_THRESHOLD,
    TARGET_SR,
    AppError,
    logger,
)

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency behavior
    AudioSegment = None


@dataclass
class AudioLoadResult:
    waveform: np.ndarray
    sr: int
    duration_sec: float
    signal_quality: str
    note: str
    rms_mean: float
    clipping_ratio: float
    uploaded_bytes: int


def _safe_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().replace(".", "")


def _clamp_duration(y: np.ndarray, sr: int) -> np.ndarray:
    max_samples = int(MAX_DURATION_SEC * sr)
    if y.shape[0] > max_samples:
        return y[:max_samples]
    return y


def _trim_silence(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    frame_len = 2048 if y.size >= 2048 else max(256, y.size)
    hop_len = max(128, frame_len // 4)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    active = np.where(rms > RMS_TRIM_THRESHOLD)[0]
    if active.size == 0:
        return y
    start_sample = max(0, active[0] * hop_len)
    end_sample = min(y.size, active[-1] * hop_len + frame_len)
    return y[start_sample:end_sample]


def _load_with_librosa(path: Path) -> np.ndarray:
    # Decode only the first MAX_DURATION_SEC to avoid long decode time / OOM on cloud runtimes.
    y, _ = librosa.load(path.as_posix(), sr=TARGET_SR, mono=True, duration=MAX_DURATION_SEC)
    return y.astype(np.float32)


def _validate_wav_header(path: Path) -> None:
    try:
        with wave.open(path.as_posix(), "rb") as wav:
            channels = wav.getnchannels()
            framerate = wav.getframerate()
            nframes = wav.getnframes()
            sample_width = wav.getsampwidth()
    except Exception as err:
        raise AppError(
            code="INVALID_WAV",
            message="WAV 파일 헤더가 손상되었거나 형식이 올바르지 않습니다.",
            hint="브라우저에서 다시 녹음하거나 표준 PCM WAV로 변환 후 업로드하세요.",
            status_code=400,
        ) from err

    if channels <= 0 or framerate <= 0 or nframes <= 0:
        raise AppError(
            code="INVALID_WAV",
            message="WAV 메타데이터가 유효하지 않습니다.",
            hint="3초 이상의 정상 WAV 파일인지 확인하세요.",
            status_code=400,
        )
    if channels > 2 or framerate > 96000 or sample_width not in {1, 2, 3, 4}:
        raise AppError(
            code="UNSUPPORTED_WAV_ENCODING",
            message="지원하지 않는 WAV 포맷입니다.",
            hint="mono/stereo PCM 16-bit WAV 파일을 업로드하세요.",
            status_code=400,
        )


def _load_wav_with_wave(path: Path) -> np.ndarray:
    try:
        with wave.open(path.as_posix(), "rb") as wav:
            channels = wav.getnchannels()
            sr_native = wav.getframerate()
            sample_width = wav.getsampwidth()
            max_frames = int(MAX_DURATION_SEC * sr_native)
            raw = wav.readframes(max_frames)
    except Exception as err:
        raise AppError(
            code="AUDIO_DECODE_ERROR",
            message="WAV 디코딩에 실패했습니다.",
            hint="손상된 파일일 수 있습니다. 파일을 다시 생성해 업로드하세요.",
            status_code=400,
        ) from err

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit PCM handling
        raw_u8 = np.frombuffer(raw, dtype=np.uint8)
        usable = (raw_u8.size // 3) * 3
        if usable == 0:
            raise AppError(
                code="INVALID_WAV",
                message="WAV 데이터 길이가 유효하지 않습니다.",
                hint="파일이 손상되었을 수 있습니다. 다시 업로드하세요.",
                status_code=400,
            )
        b = raw_u8[:usable].reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AppError(
            code="UNSUPPORTED_WAV_ENCODING",
            message="지원하지 않는 WAV 샘플 포맷입니다.",
            hint="PCM 16-bit WAV 형식으로 변환 후 업로드하세요.",
            status_code=400,
        )

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    y = np.asarray(data, dtype=np.float32)
    if sr_native != TARGET_SR:
        # Lightweight resample to reduce cloud runtime instability from heavy backends.
        old_len = y.shape[0]
        new_len = int(old_len * TARGET_SR / sr_native)
        if new_len <= 0:
            raise AppError(
                code="INVALID_AUDIO_SIGNAL",
                message="오디오 길이가 유효하지 않습니다.",
                hint="다른 파일로 다시 시도하세요.",
                status_code=400,
            )
        old_x = np.linspace(0.0, 1.0, num=old_len, endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        y = np.interp(new_x, old_x, y).astype(np.float32)
    return y.astype(np.float32)


def _convert_with_pydub(src_path: Path) -> Path:
    if AudioSegment is None:
        raise AppError(
            code="AUDIO_DECODE_ERROR",
            message="오디오 디코딩에 실패했습니다.",
            hint="pydub/ffmpeg를 설치하거나 wav 파일로 다시 업로드하세요.",
            status_code=400,
        )

    converted = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    converted_path = Path(converted.name)
    converted.close()

    try:
        audio = AudioSegment.from_file(src_path.as_posix())
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
        audio.export(converted_path.as_posix(), format="wav")
        return converted_path
    except Exception as err:
        converted_path.unlink(missing_ok=True)
        logger.exception("pydub/ffmpeg convert failed", exc_info=err)
        raise AppError(
            code="AUDIO_DECODE_ERROR",
            message="오디오 디코딩에 실패했습니다.",
            hint="Cloud 환경에서 ffmpeg가 없을 수 있습니다. wav 파일로 업로드하거나 ffmpeg를 설치하세요.",
            status_code=400,
        ) from err


def load_audio_from_upload(upload_file: UploadFile) -> AudioLoadResult:
    ext = _safe_suffix(upload_file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError(
            code="UNSUPPORTED_FORMAT",
            message="지원하지 않는 파일 형식입니다.",
            hint="wav/mp3/ogg/webm/m4a 형식 파일을 업로드하세요.",
            status_code=400,
        )

    raw_tmp = tempfile.NamedTemporaryFile(suffix=f".{ext or 'bin'}", delete=False)
    raw_path = Path(raw_tmp.name)
    raw_tmp.close()

    converted_path: Optional[Path] = None

    try:
        total_bytes = 0
        with raw_path.open("wb") as f:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise AppError(
                        code="FILE_TOO_LARGE",
                        message="업로드 파일 크기가 너무 큽니다.",
                        hint="15MB 이하 파일로 업로드하거나 녹음 길이를 줄여주세요.",
                        status_code=400,
                    )
                f.write(chunk)

        if total_bytes == 0:
            raise AppError(
                code="EMPTY_FILE",
                message="업로드 파일이 비어 있습니다.",
                hint="녹음 파일이 정상 생성되었는지 확인하세요.",
                status_code=400,
            )

        logger.info(
            "audio upload received filename=%s ext=%s bytes=%d content_type=%s",
            upload_file.filename,
            ext,
            total_bytes,
            upload_file.content_type,
        )

        try:
            if ext == "wav":
                _validate_wav_header(raw_path)
                y = _load_wav_with_wave(raw_path)
            else:
                y = _load_with_librosa(raw_path)
        except Exception as err:
            if isinstance(err, AppError):
                raise
            logger.warning("primary decode failed: %s", err)
            converted_path = _convert_with_pydub(raw_path)
            y = _load_with_librosa(converted_path)

        y = _clamp_duration(y, TARGET_SR)
        y = _trim_silence(y)
        if not np.all(np.isfinite(y)):
            raise AppError(
                code="INVALID_AUDIO_SIGNAL",
                message="오디오 신호에 유효하지 않은 값이 포함되어 있습니다.",
                hint="다른 파일로 다시 시도하세요.",
                status_code=400,
            )

        duration_sec = float(y.shape[0] / TARGET_SR) if y.size > 0 else 0.0
        if duration_sec < MIN_DURATION_SEC:
            raise AppError(
                code="AUDIO_TOO_SHORT",
                message="업로드한 음성이 너무 짧습니다.",
                hint="최소 3초 이상 녹음한 파일을 업로드하세요.",
                status_code=400,
            )

        rms_mean = float(np.sqrt(np.mean(np.square(y))) if y.size else 0.0)
        clipping_ratio = float(np.mean(np.abs(y) >= CLIPPING_VALUE) if y.size else 0.0)

        signal_quality = "good"
        notes: list[str] = []

        if rms_mean < BAD_SIGNAL_RMS_THRESHOLD:
            signal_quality = "bad"
            notes.append("입력 음량이 매우 낮습니다. 마이크 볼륨을 높여주세요.")
        elif rms_mean < BAD_SIGNAL_RMS_THRESHOLD * 2.5:
            signal_quality = "ok"
            notes.append("입력 음량이 다소 낮습니다.")

        if clipping_ratio >= CLIPPING_SAMPLE_THRESHOLD:
            if signal_quality == "good":
                signal_quality = "ok"
            notes.append("클리핑이 감지되었습니다. 녹음 게인을 낮춰주세요.")

        note = " ".join(notes) if notes else "입력 음질이 안정적입니다."

        return AudioLoadResult(
            waveform=y.astype(np.float32),
            sr=TARGET_SR,
            duration_sec=duration_sec,
            signal_quality=signal_quality,
            note=note,
            rms_mean=rms_mean,
            clipping_ratio=clipping_ratio,
            uploaded_bytes=total_bytes,
        )
    finally:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("failed to remove temp file: %s", raw_path)
        if converted_path is not None:
            try:
                converted_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("failed to remove converted temp file: %s", converted_path)
