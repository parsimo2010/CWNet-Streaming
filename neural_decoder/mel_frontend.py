"""
mel_frontend.py — Mel spectrogram frontend with SpecAugment for CW-Former.

Computes log-mel spectrograms from raw audio and applies SpecAugment
(Park et al., 2019) data augmentation during training.

Pipeline:
  Audio (16 kHz, float32)
  → STFT (25ms window, 10ms hop)
  → Mel filterbank (40 bins, 200-1400 Hz)
  → Log compression (log(mel + 1e-6))
  → SpecAugment (training only): frequency masking + time masking

Output shape: (B, T, 40).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MelFrontendConfig:
    """Configuration for the mel spectrogram frontend."""
    sample_rate: int = 16000
    n_fft: int = 400             # 25ms at 16kHz
    hop_length: int = 160        # 10ms at 16kHz
    n_mels: int = 40             # Number of mel bins
    f_min: float = 200.0         # Min frequency for mel filterbank
    f_max: float = 1400.0        # Max frequency (covers 500-900 Hz tone + filter skirts)

    # SpecAugment parameters (Park et al., 2019)
    spec_augment: bool = True    # Enable during training
    freq_mask_count: int = 2     # Number of frequency masks
    freq_mask_width: int = 8     # Max width of each frequency mask (bins)
    time_mask_count: int = 2     # Number of time masks
    time_mask_width: int = 50    # Max width of each time mask (frames)
    time_mask_ratio: float = 0.1 # Max fraction of time steps to mask


# ---------------------------------------------------------------------------
# Mel filterbank (computed once, no torchaudio dependency)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * torch.log10(torch.tensor(1.0 + hz / 700.0)).item()


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(
    n_fft: int, sample_rate: int, n_mels: int,
    f_min: float = 0.0, f_max: Optional[float] = None,
) -> Tensor:
    """Create a mel filterbank matrix, shape (n_mels, n_fft//2 + 1).

    Pure PyTorch implementation — no torchaudio dependency required.
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    # Equally spaced mel points
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # FFT bin frequencies
    fft_freqs = torch.linspace(0, sample_rate / 2.0, n_freqs)

    # Triangular filters
    filterbank = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        low = hz_points[i]
        center = hz_points[i + 1]
        high = hz_points[i + 2]

        # Rising slope
        up = (fft_freqs - low) / max(center - low, 1e-10)
        # Falling slope
        down = (high - fft_freqs) / max(high - center, 1e-10)

        filterbank[i] = torch.clamp(torch.minimum(up, down), min=0.0)

    return filterbank


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """SpecAugment data augmentation (Park et al., 2019).

    Applies frequency masking and time masking to mel spectrograms
    during training. Disabled during eval.
    """

    def __init__(self, config: MelFrontendConfig):
        super().__init__()
        self.freq_mask_count = config.freq_mask_count
        self.freq_mask_width = config.freq_mask_width
        self.time_mask_count = config.time_mask_count
        self.time_mask_width = config.time_mask_width
        self.time_mask_ratio = config.time_mask_ratio

    def forward(self, x: Tensor) -> Tensor:
        """Apply SpecAugment to mel spectrogram.

        Parameters
        ----------
        x : Tensor, shape (B, T, n_mels)

        Returns
        -------
        Tensor, same shape, with masked regions zeroed out.
        """
        if not self.training:
            return x

        B, T, F = x.shape
        x = x.clone()

        max_time_mask = min(self.time_mask_width, int(T * self.time_mask_ratio))

        for b in range(B):
            # Frequency masks
            for _ in range(self.freq_mask_count):
                f_width = random.randint(0, self.freq_mask_width)
                f_start = random.randint(0, max(0, F - f_width - 1))
                x[b, :, f_start:f_start + f_width] = 0.0

            # Time masks
            for _ in range(self.time_mask_count):
                if max_time_mask <= 0:
                    continue
                t_width = random.randint(0, max_time_mask)
                t_start = random.randint(0, max(0, T - t_width - 1))
                x[b, t_start:t_start + t_width, :] = 0.0

        return x


# ---------------------------------------------------------------------------
# Mel spectrogram frontend
# ---------------------------------------------------------------------------

class MelFrontend(nn.Module):
    """Mel spectrogram computation + optional SpecAugment.

    Computes log-mel spectrograms from raw audio waveforms using
    a pure PyTorch STFT + mel filterbank. No torchaudio dependency.

    Output shape: (B, T_frames, n_mels) where T_frames = audio_len // hop_length.
    """

    def __init__(self, config: MelFrontendConfig):
        super().__init__()
        self.config = config

        # STFT window
        self.register_buffer(
            "window",
            torch.hann_window(config.n_fft),
            persistent=False,
        )

        # Mel filterbank
        filterbank = _create_mel_filterbank(
            config.n_fft, config.sample_rate, config.n_mels,
            config.f_min, config.f_max,
        )
        self.register_buffer("mel_basis", filterbank, persistent=False)

        # SpecAugment
        self.spec_augment = SpecAugment(config) if config.spec_augment else None

    def forward(self, audio: Tensor, audio_lengths: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        """Compute log-mel spectrogram from raw audio.

        Parameters
        ----------
        audio : Tensor, shape (B, N) — raw audio waveform, float32
        audio_lengths : Tensor, shape (B,) — actual audio lengths (samples)

        Returns
        -------
        mel : Tensor, shape (B, T, n_mels) — log-mel spectrogram
        mel_lengths : Tensor, shape (B,) — actual frame counts, or None
        """
        cfg = self.config

        # STFT
        # Pad audio to avoid edge effects
        pad_amount = cfg.n_fft // 2
        audio_padded = F.pad(audio, (pad_amount, pad_amount))

        spec = torch.stft(
            audio_padded,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # Power spectrum
        power = spec.abs().pow(2)  # (B, n_fft//2+1, T)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, power)  # (B, n_mels, T)

        # Log compression
        mel = torch.log(mel + 1e-6)

        # Transpose to (B, T, n_mels)
        mel = mel.transpose(1, 2)

        # Compute output lengths
        mel_lengths = None
        T_actual = mel.shape[1]
        if audio_lengths is not None:
            # Exact STFT frame count: (padded_len - n_fft) // hop + 1
            # = (audio_len + 2*pad - n_fft) // hop + 1 = audio_len // hop + 1
            mel_lengths = audio_lengths // cfg.hop_length + 1
            mel_lengths = mel_lengths.clamp(max=T_actual)

        # SpecAugment (training only)
        if self.spec_augment is not None:
            mel = self.spec_augment(mel)

        return mel, mel_lengths

    def compute_streaming(
        self,
        audio_chunk: Tensor,
        stft_buffer: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute mel for a new audio chunk, maintaining STFT overlap state.

        For streaming inference: processes incremental audio chunks with
        overlap buffer to produce correct mel frames at chunk boundaries.
        SpecAugment is NOT applied (inference only).

        Parameters
        ----------
        audio_chunk : Tensor, shape (B, N_new) — new audio samples
        stft_buffer : Tensor, shape (B, n_fft - hop_length) = (B, 240)
            Overlap samples from previous chunk. None for first chunk.

        Returns
        -------
        mel : Tensor, shape (B, T_new, n_mels) — new mel frames
        new_buffer : Tensor, shape (B, 240) — saved for next call
        """
        cfg = self.config

        if stft_buffer is not None:
            # Prepend unconsumed samples from previous chunk
            audio = torch.cat([stft_buffer, audio_chunk], dim=-1)
        else:
            # First chunk: pad with zeros on the left (matches forward() behavior)
            audio = F.pad(audio_chunk, (cfg.n_fft // 2, 0))

        # Compute how many STFT frames fit in this audio
        audio_len = audio.shape[-1]
        n_frames = (audio_len - cfg.n_fft) // cfg.hop_length + 1 if audio_len >= cfg.n_fft else 0

        # Save unconsumed tail as buffer for next chunk.
        # The last frame starts at (n_frames-1)*hop, ends at (n_frames-1)*hop + n_fft.
        # Next frame would start at n_frames*hop. Everything from there onward is
        # unconsumed and must carry over.
        if n_frames > 0:
            consumed_up_to = n_frames * cfg.hop_length
            new_buffer = audio[:, consumed_up_to:].clone()
        else:
            # Not enough samples for even one frame — carry everything
            new_buffer = audio.clone()

        if n_frames == 0:
            empty_mel = torch.zeros(audio_chunk.shape[0], 0, cfg.n_mels,
                                    device=audio_chunk.device, dtype=audio_chunk.dtype)
            return empty_mel, new_buffer

        # STFT (only on the portion that produces complete frames)
        # Trim audio to exactly what's needed: n_frames*hop + (n_fft - hop)
        stft_len = (n_frames - 1) * cfg.hop_length + cfg.n_fft
        spec = torch.stft(
            audio[:, :stft_len],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, n_fft//2+1, T)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, power)  # (B, n_mels, T)

        # Log compression
        mel = torch.log(mel + 1e-6)

        # Transpose to (B, T, n_mels)
        mel = mel.transpose(1, 2)

        return mel, new_buffer
