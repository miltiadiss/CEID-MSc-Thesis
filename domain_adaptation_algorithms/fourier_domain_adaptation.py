import torch
import numpy as np
import librosa

def extract_ampl_phase(fft_im):
    """
    Extracts amplitude and phase from a complex-valued FFT image tensor.

    Args:
        fft_im (torch.Tensor): Complex-valued FFT tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Amplitude and phase tensors.
    """
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    """
    Replaces low-frequency components in the source amplitude with those from the target.

    Args:
        amp_src (torch.Tensor): Source amplitude tensor of shape (B, C, H, W).
        amp_trg (torch.Tensor): Target amplitude tensor of shape (B, C, H, W).
        L (float): Fraction of frequency range to replace (default is 0.1).

    Returns:
        torch.Tensor: Mutated amplitude tensor.
    """
    _, _, h, w = amp_src.shape
    b = int(np.floor(min(h, w) * L))
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    """
    Applies Fourier Domain Adaptation (FDA) by aligning low-frequency components.

    Args:
        src_img (torch.Tensor): Source image tensor of shape (B, C, H, W).
        trg_img (torch.Tensor): Target image tensor of shape (B, C, H, W).
        L (float): Fraction of frequency range to align (default is 0.1).

    Returns:
        torch.Tensor: Adapted source image in the spatial domain.
    """
    fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
    fft_trg = torch.fft.fft2(trg_img, dim=(-2, -1))
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L)
    fft_src_ = amp_src_ * torch.exp(1j * pha_src)
    return torch.fft.ifft2(fft_src_, dim=(-2, -1)).real

def wav_to_stft_tensor(y, n_fft, hop_length):
    """
    Converts a waveform to a normalized STFT magnitude tensor.

    Args:
        y (np.ndarray): Input waveform.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.

    Returns:
        Tuple[torch.Tensor, np.ndarray, int]: Normalized magnitude tensor,
        phase matrix, and original signal length.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)
    mag_norm = (mag - np.mean(mag)) / (np.std(mag) + 1e-6)
    tensor = torch.tensor(mag_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, phase, len(y)

def stft_tensor_to_wav(tensor, phase, original_len, hop_length):
    """
    Converts a normalized STFT magnitude tensor back to waveform.

    Args:
        tensor (torch.Tensor): STFT magnitude tensor.
        phase (np.ndarray): Phase matrix.
        original_len (int): Original waveform length.
        hop_length (int): Hop length.

    Returns:
        np.ndarray: Reconstructed waveform.
    """
    mag = tensor.squeeze().cpu().numpy()
    mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6)
    stft = mag * np.exp(1j * phase)
    return librosa.istft(stft, hop_length=hop_length, length=original_len)
