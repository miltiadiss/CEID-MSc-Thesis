import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

class SpectrumCorrection:
    def __init__(self, num_fft, hop_length, sub_mean=True):
        self._num_fft = num_fft
        self._hop_length = hop_length
        self._sub_mean = sub_mean
        self.coefficients = None  # {device: {class_name: coeff}}

    def _spectrum(self, audio):
        if self._sub_mean:
            audio = audio - np.mean(audio)
        stft = librosa.stft(audio, n_fft=self._num_fft, hop_length=self._hop_length)
        return np.mean(np.abs(stft), axis=-1)

    def _reduce(self, spectra, use_median, axis):
        return np.median(spectra, axis=axis) if use_median else np.mean(spectra, axis=axis)

    def fit(self, aligned_segments, reference_device, use_median=False):
        """
        Args:
            aligned_segments: list of dicts
                Each dict: { "class": class_name, device1: audio1, device2: audio2, ... }
            reference_device: str (name of reference device)
        """
        coefficients = {}

        # First collect spectra per device/class
        spectra_per_device_class = {}  # {device: {class_name: [spectra]}}
        for segment in aligned_segments:
            class_name = segment["class"]
            for device, audio in segment.items():
                if device == "class":
                    continue
                if np.isnan(audio).any():
                    continue
                spec = self._spectrum(audio)
                spectra_per_device_class \
                    .setdefault(device, {}) \
                    .setdefault(class_name, []) \
                    .append(spec)

        # Compute reference spectra (mean/median per class)
        reference_spectra = {
            class_name: self._reduce(spectra_per_device_class[reference_device][class_name], use_median, axis=0)
            for class_name in spectra_per_device_class[reference_device]
        }

        # Compute correction ratios
        coefficients = {}
        for device, class_spectra in spectra_per_device_class.items():
            if device == reference_device:
                continue
            coefficients[device] = {}
            for class_name, specs in class_spectra.items():
                device_spec = self._reduce(specs, use_median, axis=0)
                ratio = reference_spectra[class_name] / (device_spec + 1e-6)
                coefficients[device][class_name] = ratio

        self.coefficients = coefficients

    def transform_stft(self, stft, device, class_name, frequency_axis=0):
        coeffs = self.coefficients[device][class_name]
        shape = np.ones(stft.ndim, int)
        shape[frequency_axis] = -1
        return stft * coeffs.reshape(shape)

    def transform_wave(self, recording, device, class_name, return_stft=False):
        if self._sub_mean:
            recording = recording - np.mean(recording)

        stft = librosa.stft(recording, n_fft=self._num_fft, hop_length=self._hop_length)
        stft = self.transform_stft(stft, device, class_name, frequency_axis=0)

        if return_stft:
            return stft
        return librosa.istft(stft, hop_length=self._hop_length, length=len(recording))
    
    def plot_correction_coefficients(self, device_files, reference_device, k=4, sr=4000, save_dir=None):
        """
        Plot correction ratios (reference / device spectrum) for the first `k` wav files per device.

        Args:
            device_files (dict): {device: list of (filename, audio)}.
            reference_device (str): Name of reference device.
            k (int): Number of files to plot per device.
            sr (int): Sampling rate (used for x-axis in Hz).
            save_dir (str, optional): If given, saves plots to this directory.
        """
        # Find first valid reference spectrum
        ref_spec = None
        for _, audio in device_files.get(reference_device, []):
            if not np.isnan(audio).any():
                ref_spec = self._spectrum(audio)
                break
        if ref_spec is None:
            raise ValueError("No valid reference spectrum found.")

        for device, file_list in device_files.items():
            for i, (fname, audio) in enumerate(file_list[:k]):
                if np.isnan(audio).any():
                    continue

                spec = self._spectrum(audio)
                ratio = ref_spec / (spec + 1e-6)
                freqs = np.linspace(0, sr / 2, len(ratio))

                plt.figure(figsize=(10, 5))
                plt.plot(freqs, np.ones_like(ratio), label="Reference", linewidth=2)
                plt.plot(freqs, ratio, label=f"{device} | {fname}", linewidth=2)
                plt.title(f"Correction Coefficients - {device} | File {i+1}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Correction Coefficient")
                plt.legend()

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"{device}_{i}_correction.png")
                    plt.savefig(out_path)
                    plt.close()
                else:
                    plt.show()