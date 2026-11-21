"""Deterministic acoustic DSP & feature extraction matching Java AudioFeatures 1-to-1.

No librosa/soundfile dependency: uses stdlib wave + numpy + scipy.signal only.
Shared specification:
  * Target sample rate: 16,000 Hz mono
  * Windowing: N_FFT=512 (32 ms periodic Hann), HOP=320 (20 ms), N_FRAMES=128 (~2.572 s)
  * Mel filterbank: 64 HTK triangular filters, 50 Hz .. 7600 Hz
  * Log-Mel: ln(Mel + 1e-6)
  * MFCC: 20 orthogonal DCT-II coefficients of 64-bin Log-Mel + temporal delta
  * Functionals: 64-dim utterance summary vector (MFCC mean/std, Chroma, F0, RMS, ZCR, Spectral)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 320
N_FRAMES = 128
N_MELS = 64
N_MFCC = 20
TARGET_SAMPLES = (N_FRAMES - 1) * HOP_LENGTH + N_FFT  # 41152 samples = 2.572 s
F_MIN = 50.0
F_MAX = 7600.0


def _hann_periodic(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / n))


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """HTK triangular Mel filterbank [n_mels, n_fft//2 + 1] with area normalization."""
    n_freqs = n_fft // 2 + 1
    freqs = np.arange(n_freqs, dtype=np.float64) * (sr / n_fft)
    m_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = _mel_to_hz(m_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        f_left, f_center, f_right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        lower = (freqs - f_left) / max(1e-12, f_center - f_left)
        upper = (f_right - freqs) / max(1e-12, f_right - f_center)
        tri = np.maximum(0.0, np.minimum(lower, upper))
        enorm = 2.0 / max(1e-12, f_right - f_left)
        fb[m] = tri * enorm
    return fb


def build_chroma_matrix(sr: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """Maps STFT bins (80..2500 Hz) to 12 pitch-class chroma weights."""
    n_freqs = n_fft // 2 + 1
    mat = np.zeros((12, n_freqs), dtype=np.float64)
    for k in range(1, n_freqs):
        f = k * sr / n_fft
        if 80.0 <= f <= 2500.0:
            midi = 69.0 + 12.0 * math.log2(f / 440.0)
            pc = int(round(midi)) % 12
            mat[pc, k] = 1.0
    col_sum = mat.sum(axis=1, keepdims=True)
    col_sum[col_sum == 0.0] = 1.0
    return mat / col_sum


_HANN = _hann_periodic(N_FFT)
_MEL_FB = build_mel_filterbank()
_CHROMA_FB = build_chroma_matrix()


@dataclass
class AudioValidation:
    sr: int
    channels: int
    raw_duration_s: float
    peak: float
    clipping_ratio: float
    dc_offset: float
    silence_ratio: float
    snr_db: float
    valid: bool
    reason: str


def load_and_validate_wav(path: Path) -> tuple[np.ndarray, AudioValidation]:
    """Reads WAV, converts to 16 kHz mono float64, and computes quality diagnostics."""
    try:
        sr, data = wavfile.read(str(path))
    except Exception as exc:  # noqa: BLE001
        return np.zeros(0, dtype=np.float64), AudioValidation(
            0, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, f"decode_error:{exc}"
        )

    channels = 1 if data.ndim == 1 else data.shape[1]
    if data.size == 0:
        return np.zeros(0, dtype=np.float64), AudioValidation(
            sr, channels, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, "empty"
        )

    if np.issubdtype(data.dtype, np.integer):
        max_int = float(np.iinfo(data.dtype).max)
        y = data.astype(np.float64) / max_int
    else:
        y = data.astype(np.float64)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if not np.all(np.isfinite(y)):
        return np.zeros(0, dtype=np.float64), AudioValidation(
            sr, channels, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, "nan_or_inf"
        )

    raw_duration = len(y) / float(sr)
    dc_offset = float(np.mean(y))
    y = y - dc_offset
    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    clipping_ratio = float(np.mean(np.abs(y) >= 0.995))

    if sr != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // g, int(sr) // g)

    # Frame-level energy for silence ratio and SNR estimation (20 ms non-overlapping)
    step = HOP_LENGTH
    n_f = max(1, (len(y) - step) // step)
    frame_rms = np.array([
        math.sqrt(float(np.mean(y[i * step : i * step + step] ** 2)) + 1e-12)
        for i in range(n_f)
    ])
    noise_floor = float(np.percentile(frame_rms, 10))
    speech_rms = float(np.percentile(frame_rms, 90))
    snr_db = 20.0 * math.log10(max(speech_rms, 1e-9) / max(noise_floor, 1e-9))
    silence_thresh = max(1e-4, 0.1 * float(np.max(frame_rms)))
    silence_ratio = float(np.mean(frame_rms < silence_thresh))

    valid = (raw_duration >= 0.35) and (peak >= 1e-4) and (silence_ratio < 0.98)
    reason = "ok" if valid else f"dur={raw_duration:.2f},peak={peak:.5f},sil={silence_ratio:.2f}"
    return y, AudioValidation(
        sr=int(sr),
        channels=int(channels),
        raw_duration_s=raw_duration,
        peak=peak,
        clipping_ratio=clipping_ratio,
        dc_offset=dc_offset,
        silence_ratio=silence_ratio,
        snr_db=snr_db,
        valid=valid,
        reason=reason,
    )


def preprocess_signal(y_16k: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Trims leading/trailing silence, records pre-norm RMS, normalizes peak to 0.95, and pads/crops."""
    if len(y_16k) == 0:
        return np.zeros(TARGET_SAMPLES, dtype=np.float64), 0.0, -90.0

    # Simple energy VAD trim with 2-frame (40 ms) collar
    step = HOP_LENGTH
    n_f = max(1, len(y_16k) // step)
    energies = np.array([
        float(np.mean(y_16k[i * step : (i + 1) * step] ** 2))
        for i in range(n_f)
    ])
    thresh = max(1e-8, 0.01 * float(np.max(energies)))
    active = np.where(energies >= thresh)[0]
    if len(active) > 0:
        start_f = max(0, int(active[0]) - 2)
        end_f = min(n_f, int(active[-1]) + 3)
        trimmed = y_16k[start_f * step : end_f * step]
        if len(trimmed) >= int(0.4 * SAMPLE_RATE):
            y_16k = trimmed

    pre_rms = math.sqrt(float(np.mean(y_16k ** 2)) + 1e-12)
    pre_rms_db = 20.0 * math.log10(max(pre_rms, 1e-9))

    peak = float(np.max(np.abs(y_16k)))
    if peak > 1e-6:
        y_norm = y_16k * (0.95 / peak)
    else:
        y_norm = y_16k.copy()

    if len(y_norm) > TARGET_SAMPLES:
        mid = len(y_norm) // 2
        half = TARGET_SAMPLES // 2
        fixed = y_norm[mid - half : mid - half + TARGET_SAMPLES]
    elif len(y_norm) < TARGET_SAMPLES:
        fixed = np.zeros(TARGET_SAMPLES, dtype=np.float64)
        fixed[: len(y_norm)] = y_norm
    else:
        fixed = y_norm

    return fixed, pre_rms, pre_rms_db


def stft_power(signal_fixed: np.ndarray) -> np.ndarray:
    """Uncentered Hann-windowed STFT power spectrum [257, 128]."""
    idx = np.arange(N_FRAMES)[:, None] * HOP_LENGTH + np.arange(N_FFT)[None, :]
    frames = signal_fixed[idx] * _HANN[None, :]
    spec = np.fft.rfft(frames, axis=1)  # [128, 257]
    power = (spec.real ** 2 + spec.imag ** 2).T  # [257, 128]
    return power


def estimate_f0_autocorr(signal_fixed: np.ndarray) -> tuple[float, float, float]:
    """Normalized autocorrelation F0 estimator across frames (70..450 Hz)."""
    min_lag = SAMPLE_RATE // 450  # 35
    max_lag = SAMPLE_RATE // 70   # 228
    f0s = []
    voiced_count = 0
    total_frames = 0
    for t in range(N_FRAMES):
        start = t * HOP_LENGTH
        frame = signal_fixed[start : start + N_FFT]
        energy = float(np.mean(frame ** 2))
        total_frames += 1
        if energy < 1e-4:
            continue
        frame_z = frame - np.mean(frame)
        denom = float(np.sum(frame_z ** 2))
        if denom < 1e-8:
            continue
        best_r = 0.0
        best_lag = 0
        for lag in range(min_lag, max_lag + 1):
            num = float(np.dot(frame_z[:-lag], frame_z[lag:]))
            r = num / denom
            if r > best_r:
                best_r = r
                best_lag = lag
        if best_r > 0.35 and best_lag > 0:
            f0s.append(SAMPLE_RATE / float(best_lag))
            voiced_count += 1
    if not f0s:
        return 0.0, 0.0, 0.0
    arr = np.asarray(f0s, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), voiced_count / max(1, total_frames)


def extract_all_features(y_16k: np.ndarray) -> dict[str, np.ndarray]:
    """Extracts matched Log-Mel [64, 128], MFCC+Delta [40, 128], STFT [128, 128], and 64-d Functionals."""
    fixed, pre_rms, pre_rms_db = preprocess_signal(y_16k)
    power = stft_power(fixed)  # [257, 128]
    mel = _MEL_FB @ power      # [64, 128]
    log_mel = np.log(mel + 1e-6)

    # Orthogonal DCT-II across Mel axis -> 20 MFCCs [20, 128]
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    delta = np.zeros_like(mfcc)
    delta[:, 1:-1] = 0.5 * (mfcc[:, 2:] - mfcc[:, :-2])
    delta[:, 0] = mfcc[:, 1] - mfcc[:, 0]
    delta[:, -1] = mfcc[:, -1] - mfcc[:, -2]
    mfcc_delta = np.concatenate([mfcc, delta], axis=0)  # [40, 128]

    # Active frames mask (avoid zero-padding skewing functionals)
    frame_energy = np.array([
        float(np.mean(fixed[t * HOP_LENGTH : t * HOP_LENGTH + N_FFT] ** 2))
        for t in range(N_FRAMES)
    ])
    active_mask = frame_energy > max(1e-7, 0.005 * float(np.max(frame_energy)))
    if not np.any(active_mask):
        active_mask = np.ones(N_FRAMES, dtype=bool)

    mfcc_act = mfcc[:, active_mask]
    mfcc_mean = np.mean(mfcc_act, axis=1)
    mfcc_std = np.std(mfcc_act, axis=1)

    chroma = _CHROMA_FB @ power[:, active_mask]
    chroma_mean = np.mean(chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8), axis=1)

    f0_mean, f0_std, voiced_ratio = estimate_f0_autocorr(fixed)

    rms_frames = np.sqrt(frame_energy[active_mask] + 1e-12)
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))
    rms_max = float(np.max(rms_frames))

    zcr_vals = np.array([
        float(np.mean(np.abs(np.diff(np.signbit(fixed[t * HOP_LENGTH : t * HOP_LENGTH + N_FFT])))))
        for t in np.where(active_mask)[0]
    ])
    zcr_mean = float(np.mean(zcr_vals))
    zcr_std = float(np.std(zcr_vals))

    mag = np.sqrt(power[:, active_mask] + 1e-12)
    freqs = np.arange(N_FFT // 2 + 1, dtype=np.float64) * (SAMPLE_RATE / N_FFT)
    mag_sum = np.sum(mag, axis=0) + 1e-8
    centroid_khz = float(np.mean((freqs @ mag) / mag_sum)) / 1000.0
    cum_mag = np.cumsum(mag, axis=0) / mag_sum[None, :]
    rolloff_idx = np.argmax(cum_mag >= 0.85, axis=0)
    rolloff_khz = float(np.mean(freqs[rolloff_idx])) / 1000.0
    flux = float(np.mean(np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0)))) if mag.shape[1] > 1 else 0.0

    functionals = np.concatenate([
        mfcc_mean,                                                      # 20
        mfcc_std,                                                       # 20
        chroma_mean,                                                    # 12
        np.array([
            f0_mean / 300.0, f0_std / 100.0, voiced_ratio,              # 3
            pre_rms, pre_rms_db / 60.0, rms_mean, rms_std, rms_max,     # 5 (wait: total=60; +4 = 64)
            zcr_mean, zcr_std, centroid_khz, rolloff_khz                # 4 -> 64!
        ], dtype=np.float64),
    ])

    # Downsampled 64-bin linear STFT log magnitude [64, 128] for E1 ablation
    stft_64 = np.log(np.sqrt(power[1:257].reshape(64, 4, N_FRAMES).mean(axis=1)) + 1e-6)

    return {
        "log_mel": log_mel.astype(np.float32),
        "mel_linear": np.log1p(mel).astype(np.float32),
        "stft_64": stft_64.astype(np.float32),
        "mfcc_delta": mfcc_delta.astype(np.float32),
        "functionals": functionals.astype(np.float32),
        "pre_rms": np.float32(pre_rms),
        "f0_mean": np.float32(f0_mean),
        "flux": np.float32(flux),
    }
