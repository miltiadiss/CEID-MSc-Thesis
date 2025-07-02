import numpy as np
import librosa
import os
import soundfile as sf
from tqdm import tqdm

class ClassWiseSpectralWhitening:
    """
    Class-wise spectral whitening for domain adaptation.
    Transforms signals so that their power spectrum matches the reference class spectrum.
    """

    def __init__(self, n_fft=256, hop_length=64, epsilon=1e-6):
        """
        Args:
            n_fft (int): FFT window size.
            hop_length (int): Hop length.
            epsilon (float): Small value to avoid division by zero.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.epsilon = epsilon
        self.stats = {}  # {class_name: mean power spectrum (freq_bins,)}

    def _compute_power_spectrum(self, y):
        """
        Compute mean power spectrum over time for a signal.
        """
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        power = np.abs(stft) ** 2
        mean_power = np.mean(power, axis=1)  # mean over time
        return mean_power

    def fit(self, file_list, labels, devices, reference_device):
        """
        Fit mean power spectrum per class from reference device.

        Args:
            file_list (list): List of file paths.
            labels (list): List of class names.
            devices (list): List of device names.
            reference_device (str): Name of the reference device.
        """
        spectra = {}  # {class_name: list of mean power spectra}

        for file, label, device in tqdm(zip(file_list, labels, devices), total=len(file_list), desc="Fitting whitening"):
            if reference_device not in device:
                continue
            y, _ = librosa.load(file, sr=None)
            mean_power = self._compute_power_spectrum(y)
            spectra.setdefault(label, []).append(mean_power)

        for label, spec_list in spectra.items():
            self.stats[label] = np.mean(spec_list, axis=0)

    def transform(self, y, class_name):
        """
        Apply spectral whitening to signal.

        Args:
            y (np.ndarray): Waveform.
            class_name (str): Class of the signal.

        Returns:
            np.ndarray: Spectral whitened waveform.
        """
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(stft)
        phase = np.angle(stft)
        mean_power = self.stats[class_name]
        corrected_mag = mag / (np.sqrt(mean_power[:, np.newaxis]) + self.epsilon)
        stft_corrected = corrected_mag * np.exp(1j * phase)
        y_out = librosa.istft(stft_corrected, hop_length=self.hop_length, length=len(y))
        return y_out