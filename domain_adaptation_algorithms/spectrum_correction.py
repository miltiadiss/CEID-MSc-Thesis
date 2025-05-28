import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

class SpectrumCorrection:
    """Applies spectrum correction based on average STFT spectra from aligned segments.

    Args:
        num_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        sub_mean (bool): Whether to subtract the mean before STFT.
    """

    def __init__(self, num_fft, hop_length, sub_mean=True):
        self._num_fft = num_fft
        self._hop_length = hop_length
        self._sub_mean = sub_mean
        self.coefficients = None

    def _spectrum(self, audio):
        if self._sub_mean:
            audio -= np.mean(audio)
        stft = librosa.stft(audio, n_fft=self._num_fft, hop_length=self._hop_length)
        return np.mean(np.abs(stft), axis=-1)

    def _reduce(self, spectra, use_median, axis):
        return np.median(spectra, axis=axis) if use_median else np.mean(spectra, axis=axis)

    def fit(self, aligned_segments, reference, use_median=False):
        """Fit the correction coefficients from aligned segments.

        Args:
            aligned_segments (list[dict]): Each element contains {device: audio} for aligned recordings.
            reference (str or list): Reference device(s) to normalize others against.
            use_median (bool): Use median instead of mean for coefficient aggregation.
        """
        coefficients = {}

        for segment in aligned_segments:
            spectra = {
                device: self._spectrum(audio)
                for device, audio in segment.items()
                if not np.isnan(audio).any()
            }

            base_spec = self._reference_spectrum(reference, spectra)

            for device, spec in spectra.items():
                ratio = base_spec / (spec + 1e-6)
                coefficients.setdefault(device, []).append(ratio)

        self.coefficients = {
            device: self._reduce(specs, use_median, axis=0)
            for device, specs in coefficients.items()
        }

    def _reference_spectrum(self, reference, spectra):
        if isinstance(reference, str):
            return spectra[reference]
        elif isinstance(reference, list):
            return np.mean([spectra[d] for d in reference if d in spectra], axis=0)
        else:
            raise ValueError("Invalid reference specification.")

    def transform_stft(self, stft, device, frequency_axis=0):
        """Apply spectrum correction directly to STFT."""
        shape = np.ones(stft.ndim, int)
        shape[frequency_axis] = -1
        return stft * self.coefficients[device].reshape(shape)

    def transform_wave(self, recording, device, return_stft=False):
        """Apply correction to time-domain signal."""
        if self._sub_mean:
            recording = recording - np.mean(recording)

        stft = librosa.stft(recording, n_fft=self._num_fft, hop_length=self._hop_length)
        stft = self.transform_stft(stft, device, frequency_axis=0)

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
                plt.grid(True)
                plt.legend()

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"{device}_{i}_correction.png")
                    plt.savefig(out_path)
                    plt.close()
                else:
                    plt.show()