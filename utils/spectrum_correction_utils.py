import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.spatial.distance import euclidean

# === Functions ===
def compute_average_spectrum(signals: List[np.ndarray], sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Compute the geometric mean spectrum over a list of signals.

    Args:
        signals (List[np.ndarray]): List of audio signals.
        sr (int): Sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop size.

    Returns:
        np.ndarray: Mean spectrum (1D array of shape [n_fft//2+1]).
    """
    spectra = []
    for y in signals:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-8
        mean_spec = np.mean(S, axis=1)
        spectra.append(mean_spec)
    stacked = np.stack(spectra, axis=0)
    return np.exp(np.mean(np.log(stacked), axis=0))

def compute_all_reference_spectra(signals: List[np.ndarray], sr: int, n_fft: int, hop_length: int) -> List[np.ndarray]:
    """Compute individual average spectra for each signal.

    Args:
        signals (List[np.ndarray]): List of signals.
        sr (int): Sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop size.

    Returns:
        List[np.ndarray]: List of 1D spectra.
    """
    return [np.mean(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-8, axis=1) for y in signals]

def find_closest_reference_spectrum(target_spectrum: np.ndarray, ref_spectra: List[np.ndarray]) -> np.ndarray:
    """Find the reference spectrum most similar to the target.

    Args:
        target_spectrum (np.ndarray): Spectrum to match.
        ref_spectra (List[np.ndarray]): List of reference spectra.

    Returns:
        np.ndarray: Closest matching reference spectrum.
    """
    return min(ref_spectra, key=lambda ref: euclidean(ref, target_spectrum))

def compute_correction_coefficients(
    source_spectrum: np.ndarray,
    target_spectrum: np.ndarray
) -> np.ndarray:
    """Compute correction coefficients to map one device spectrum to another.

    Args:
        source_spectrum (np.ndarray): Mean spectrum of the source device.
        target_spectrum (np.ndarray): Mean spectrum of the reference device.

    Returns:
        np.ndarray: Correction coefficients for each frequency bin.
    """
    return target_spectrum / (source_spectrum + 1e-8)

def apply_spectrum_correction(
    y: np.ndarray,
    sr: int,
    correction_coeffs: np.ndarray,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """Apply spectrum correction to a signal in the frequency domain.

    Args:
        y (np.ndarray): Input audio signal.
        sr (int): Sampling rate.
        correction_coeffs (np.ndarray): Frequency-wise correction coefficients.
        n_fft (int): FFT window size.
        hop_length (int): Hop size.

    Returns:
        np.ndarray: Spectrum corrected audio signal.
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)
    corrected_mag = correction_coeffs[:, np.newaxis] * magnitude
    D_corrected = corrected_mag * np.exp(1j * phase)
    y_corrected = librosa.istft(D_corrected, hop_length=hop_length)
    return y_corrected

def get_signals_by_device(folder: str, device: str, sr: int) -> Tuple[List[np.ndarray], List[str]]:
    """Load all signals and filenames for a specific recording device.

    Args:
        folder (str): Path to folder containing .wav files.
        device (str): Device name substring to filter.
        sr (int): Target sampling rate.

    Returns:
        Tuple[List[np.ndarray], List[str]]: List of signals and corresponding filenames.
    """
    signals = []
    filenames = []
    for file in os.listdir(folder):
        if file.endswith(".wav") and device in file:
            path = os.path.join(folder, file)
            y, _ = librosa.load(path, sr=sr)
            signals.append(y)
            filenames.append(file)
    return signals, filenames
