import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.signal import butter, filtfilt, correlate
import scipy.stats
from typing import Tuple

# === Functions ===
def highpass_filter(y: np.ndarray, cutoff: float, sr: int, order: int = 5) -> np.ndarray:
    """Apply high-pass Butterworth filter.

    Args:
        y (np.ndarray): Input audio signal.
        cutoff (float): Cutoff frequency (Hz).
        sr (int): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered audio signal.
    """

    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='highpass', analog=False)
    return filtfilt(b, a, y)

def preprocess_audio(audio_path: str, sr: int, cutoff: float = 50.0) -> Tuple[np.ndarray, int]:
    """Load, filter and normalize audio files.

    Args:
        audio_path (str): Path to audio (.wav).
        sr (int): Sampling rate.
        cutoff (float): High-pass filter cutoff frequency.

    Returns:
        tuple: Processed audio signal (np.array), sampling rate (int).
    """
    y, sr = librosa.load(audio_path, sr=sr)
    y = highpass_filter(y, cutoff, sr, order=5)
    y = librosa.util.normalize(y)
    return y, sr

def fix_cycle_length(y: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Pad or truncate signal to fixed length.

    Args:
        y (np.ndarray): Input audio.
        sr (int): Sampling rate.
        target_duration (float): Desired duration in seconds.

    Returns:
        np.ndarray: Signal of fixed length.
    """
    target_length = int(sr * target_duration)
    if len(y) < target_length:
        repeat_times = int(np.ceil(target_length / len(y)))
        y = np.tile(y, repeat_times)[:target_length]
    else:
        y = y[:target_length]
    return y

def extract_acoustic_features(
    y: np.ndarray,
    sr: int,
    frame_size: int,
    hop_size: int,
    n_mfcc: int,
    n_mels: int,
) -> np.ndarray:
    """Extract acoustic features (MFCC, GFCC, temporal, spectral and Log-Mel statistical features).

    Args:
        y (np.ndarray): Input audio signal.
        sr (int): Sampling rate.
        frame_size (int): Number of samples per frame.
        hop_size (int): Number of samples per hop.
        n_mfcc (int): Number of MFCC coefficients.
        n_mels (int): Number of Mel filters.

    Returns:
        np.ndarray: 1D feature vector.
    """

    # MFCCs
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_size,
        hop_length=hop_size, window='hann'
    )

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size, hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_size, hop_length=hop_size)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_size, hop_length=hop_size)

    # Temporal features
    rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_size, hop_length=hop_size)

    # Spectral flux (from centroid)
    spectral_flux = np.diff(spectral_centroid, axis=1)
    spectral_flux = np.hstack([spectral_flux, spectral_flux[:, -1:]])

    # Log-mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_size, hop_length=hop_size,
                                                      n_mels=n_mels, window='hann')
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    log_mel_mean = np.mean(log_mel, axis=1)
    log_mel_std = np.std(log_mel, axis=1)
    log_mel_skewness = scipy.stats.skew(log_mel, axis=1, bias=False)
    log_mel_kurtosis = scipy.stats.kurtosis(log_mel, axis=1, bias=False)

    # Autocorrelation peak
    autocorr = correlate(y, y, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep only positive lags
    autocorr_peak = np.max(autocorr[1:])  # exclude zero-lag

    # Temporal entropy (based on RMS energy)
    rms_energy = rms[0] / (np.sum(rms[0]) + 1e-8)
    temporal_entropy = -np.sum(rms_energy * np.log2(rms_energy + 1e-8))

    # Final feature vector
    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_centroid), np.mean(spectral_bandwidth),
        np.mean(zero_crossing_rate), np.mean(rms),
        np.mean(spectral_rolloff), np.mean(spectral_flux),
        np.mean(spectral_flatness),
        np.mean(log_mel_mean), np.mean(log_mel_std),
        np.mean(log_mel_skewness), np.mean(log_mel_kurtosis),
        autocorr_peak,
        temporal_entropy
    ])

    return feature_vector

def apply_vtlp(
    y: np.ndarray,
    sr: int,
    frame_size: int,
    hop_size: int,
    n_mels: int,
    f_hi: float,
    alpha: float = 1.0
) -> np.ndarray:
    """Apply Vocal Tract Length Perturbation (VTLP) for data augmentation.

    Args:
        y (np.ndarray): Input signal.
        sr (int): Sampling rate.
        frame_size (int): Number of samples per frame.
        hop_size (int): Number of samples per hop.
        n_mels (int): Number of Mel filters.
        f_hi (float): Upper boundary frequency (e.g. 2000 Hz).
        alpha (float): Warping factor.

    Returns:
        np.ndarray: VTLP-augmented signal.
    """

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_size, hop_length=hop_size, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)

    warped_freqs = []
    for f in mel_freqs:
        if f <= f_hi / alpha:
            warped_freqs.append(alpha * f)
        else:
            warped = (sr / 2 - f_hi) / (sr / 2 - (f_hi / alpha)) * (f - (f_hi / alpha)) + f_hi
            warped_freqs.append(warped)

    warped_log_mel = np.zeros_like(log_mel)
    for t in range(log_mel.shape[1]):
        warped_log_mel[:, t] = np.interp(mel_freqs, warped_freqs, log_mel[:, t])

    noise = np.random.normal(0, 0.01, size=warped_log_mel.shape)
    warped_log_mel += noise

    power_mel = librosa.db_to_power(warped_log_mel)
    y_warped = librosa.feature.inverse.mel_to_audio(power_mel, sr=sr, n_fft=frame_size, hop_length=hop_size)
    return y_warped

def classify(crackle: int, wheeze: int) -> str:
    """Assign a class label based on crackle and wheeze presence.

    Args:
        crackle (int): 1 if crackle present, 0 otherwise.
        wheeze (int): 1 if wheeze present, 0 otherwise.

    Returns:
        str: One of {'Normal', 'Crackle', 'Wheeze', 'Both'}.
    """

    if crackle == 0 and wheeze == 0: return "Normal"
    if crackle == 1 and wheeze == 0: return "Crackle"
    if crackle == 0 and wheeze == 1: return "Wheeze"
    return "Both"