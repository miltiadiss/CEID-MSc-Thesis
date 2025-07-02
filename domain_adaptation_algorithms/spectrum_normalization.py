import numpy as np
import librosa

class ClassWiseSpectralStandardizer:
    """Performs class-wise spectral standardization based on a reference device."""
    
    def __init__(self, n_fft=256, hop_length=64, epsilon=1e-6):
        """
        Initialize the standardizer.
        
        Args:
            n_fft (int): FFT window size.
            hop_length (int): Hop length for STFT.
            epsilon (float): Small value to avoid division by zero.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.epsilon = epsilon
        self.stats = {}  # {class_name: {"mean": array, "std": array}}

    def _compute_spectrum(self, y):
        """
        Compute the magnitude spectrum of an audio signal.
        
        Args:
            y (np.ndarray): Waveform signal.
        
        Returns:
            np.ndarray: Magnitude spectrum (frequency bins x time frames).
        """
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(stft)
        return mag

    def fit(self, file_list, labels, devices, reference_device):
        """
        Compute mean and std per class on the reference device.
        
        Args:
            file_list (list): List of .wav file paths.
            labels (list): List of class labels.
            devices (list): List of device names corresponding to each file.
            reference_device (str): Name of the reference device.
        """
        spectra_by_class = {}

        for fname, label, device in zip(file_list, labels, devices):
            if reference_device not in device:
                continue
            y, _ = librosa.load(fname, sr=None)
            mag = self._compute_spectrum(y)
            if label not in spectra_by_class:
                spectra_by_class[label] = []
            spectra_by_class[label].append(mag)

        for label, specs in spectra_by_class.items():
            specs = np.stack([np.mean(s, axis=1) for s in specs])  # mean over time
            mean = np.mean(specs, axis=0)
            std = np.std(specs, axis=0)
            self.stats[label] = {"mean": mean, "std": std}
    
    def transform(self, y, class_name):
        """
        Normalize the spectrum of a signal based on reference stats of its class.
        
        Args:
            y (np.ndarray): Waveform signal.
            class_name (str): Class label of the signal.
        
        Returns:
            np.ndarray: Normalized spectrum (frequency bins x time frames).
        """
        mag = self._compute_spectrum(y)
        mean = self.stats[class_name]["mean"]
        std = self.stats[class_name]["std"]
        norm_spec = (mag - mean[:, np.newaxis]) / (std[:, np.newaxis] + self.epsilon)
        return norm_spec

    def transform_inverse(self, norm_spec, original_phase, length):
        """
        Reconstruct waveform from normalized spectrum and original phase.
        
        Args:
            norm_spec (np.ndarray): Normalized spectrum.
            original_phase (np.ndarray): Phase matrix from original STFT.
            length (int): Original signal length.
        
        Returns:
            np.ndarray: Reconstructed waveform.
        """
        stft = norm_spec * np.exp(1j * original_phase)
        y = librosa.istft(stft, hop_length=self.hop_length, length=length)
        return y
