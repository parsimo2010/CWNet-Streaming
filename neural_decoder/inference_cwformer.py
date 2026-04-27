"""
inference_cwformer.py — Streaming causal inference for the CW-Former.

Processes audio chunk-by-chunk with state carry-forward (KV cache + conv
buffers). No windowing, no stitching. Characters are emitted immediately
as each chunk is decoded — causal CTC greedy decode guarantees that past
output never changes, so there is no reason to delay emission.

Usage (Python API):
    from neural_decoder.inference_cwformer import CWFormerStreamingDecoder

    dec = CWFormerStreamingDecoder("checkpoints_cwformer/best_model.pt")
    text = dec.decode_file("morse.wav")

    # Or streaming:
    dec.reset()
    for audio_chunk in audio_source:
        new_chars = dec.feed_audio(audio_chunk)
        print(new_chars, end="", flush=True)
    final = dec.flush()

Usage (CLI):
    python -m neural_decoder.inference_cwformer \\
        --checkpoint checkpoints_cwformer/best_model.pt \\
        --input morse.wav
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import vocab as vocab_module
from neural_decoder.conformer import ConformerConfig
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.mel_frontend import MelFrontendConfig


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_cwformer_checkpoint(
    checkpoint: str,
    device: torch.device,
) -> Tuple[CWFormer, CWFormerConfig, int]:
    """Load CW-Former checkpoint.

    Returns (model, config, sample_rate).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    mc = ckpt.get("model_config", {})

    mel_cfg = MelFrontendConfig(
        sample_rate=mc.get("sample_rate", 16000),
        n_mels=mc.get("n_mels", 40),
        n_fft=mc.get("n_fft", 400),
        hop_length=mc.get("hop_length", 160),
        f_min=mc.get("f_min", 200.0),
        f_max=mc.get("f_max", 1400.0),
        spec_augment=False,  # no augmentation at inference
    )
    conformer_cfg = ConformerConfig(
        d_model=mc.get("d_model", 256),
        n_heads=mc.get("n_heads", 4),
        n_layers=mc.get("n_layers", 12),
        d_ff=mc.get("d_ff", 1024),
        conv_kernel=mc.get("conv_kernel", 63),
        dropout=0.0,  # no dropout at inference
        # Old checkpoints saved 1475 (the pre-SWA architectural cap).
        # New checkpoints save 250 (matching the inference-time cache cap
        # and the training-time sliding-window). Either loads correctly;
        # the streaming wrapper writes through the runtime override on top.
        max_cache_len=mc.get("max_cache_len", 250),
    )
    model_cfg = CWFormerConfig(
        mel=mel_cfg, conformer=conformer_cfg,
    )

    model = CWFormer(model_cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    return model, model_cfg, mel_cfg.sample_rate


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, return float32 mono."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        import torchaudio
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)
        audio = audio_t.squeeze(0).numpy()
    return audio


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
    """Scale audio so that its peak magnitude equals target_peak.

    Matches the peak-normalization morse_generator applies to every
    training sample. Without it, the log-mel feature scale at the model
    input depends on the caller's recording gain, shifting subsampling
    ReLU outputs off-distribution.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-9:
        return (audio * (target_peak / peak)).astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# CWFormerStreamingDecoder
# ---------------------------------------------------------------------------

class CWFormerStreamingDecoder:
    """Streaming causal decoder for CW-Former.

    Processes audio chunk-by-chunk with state carry-forward.
    No windowing. No stitching. Characters emitted immediately
    after each chunk is processed.

    Because the model is fully causal, each frame's CTC output is
    determined solely by past audio and never changes. Greedy CTC
    decode of a prefix is always a correct prefix of the full decode.
    There is no need for a commitment delay — characters are emitted
    as soon as they are decoded.

    Latency = chunk accumulation time + model processing time.
    With chunk_ms=500: ~500ms + ~30ms = ~530ms typical.

    Args:
        checkpoint: Path to CW-Former checkpoint.
        chunk_ms: Audio chunk size in milliseconds (default 500).
            Smaller = lower latency, more frequent updates.
            Larger = slightly better throughput (fewer Python/CUDA overheads).
            Does not affect accuracy (mathematically identical output).
        device: PyTorch device string.
        max_cache_sec: Maximum KV cache duration in seconds (default 5).
            Empirically, holding more than ~5 s of state across QSO
            handoffs (fist + pitch + AGC change) causes the model to lock
            onto the previous operator's tokens and miss the new one. The
            rolling cache cap forces the prior context to roll off naturally.
            Training audio went up to 30 s, but multi-segment training
            already taught the model to function from short segments, so
            5 s of attention context is plenty for letter-level decoding.
            Set to None to use the cache length stored in the checkpoint
            config.
        blank_trim_sec: If the rolling greedy-argmax window of this many
            seconds contains only blank (idx 0) and space (idx 1) frames,
            actively flush the encoder state (KV cache, conv buffers,
            sub buffers, pos_offset). Mel STFT overlap is preserved so
            the next chunk's spectrogram remains continuous. Re-arms only
            after a non-blank-non-space frame is observed, so each silent
            stretch fires the reset at most once. Default 5.0 seconds.
            Pass None or a non-positive value to disable.
    """

    def __init__(
        self,
        checkpoint: str,
        chunk_ms: int = 500,
        device: str = "cpu",
        max_cache_sec: Optional[float] = 5.0,
        blank_trim_sec: Optional[float] = 5.0,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_ms = chunk_ms

        self._model, self._model_cfg, self.sample_rate = (
            _load_cwformer_checkpoint(checkpoint, self.device)
        )

        # Cap KV cache to match training distribution (relative positions
        # exceeding training's max are RoPE extrapolation territory).
        # CTC output frames are at 50 fps (2× subsampling from 10 ms mel).
        frames_per_sec = self.sample_rate // self._model_cfg.mel.hop_length // 2
        if max_cache_sec is not None:
            self._model.config.conformer.max_cache_len = int(max_cache_sec * frames_per_sec)

        # Silence-triggered KV cache reset.
        if blank_trim_sec is not None and blank_trim_sec > 0:
            self._blank_trim_frames = int(blank_trim_sec * frames_per_sec)
        else:
            self._blank_trim_frames = 0

        # Chunk size in samples
        self._chunk_samples = int(chunk_ms * self.sample_rate / 1000)

        # Internal state
        self.reset()

    @classmethod
    def from_model(
        cls,
        model: CWFormer,
        model_cfg: CWFormerConfig,
        sample_rate: int,
        chunk_ms: int = 500,
        device: Optional[torch.device] = None,
        max_cache_sec: Optional[float] = 5.0,
        blank_trim_sec: Optional[float] = 5.0,
    ) -> "CWFormerStreamingDecoder":
        """Build a decoder around an already-instantiated model.

        Intended for in-training use (streaming-mode validation pass) where
        loading from a checkpoint file is unnecessary. The passed-in model
        is used in place — the caller must set ``model.eval()`` before
        running validation.

        ``max_cache_sec`` writes through to ``model.config.conformer.
        max_cache_len``; this only affects ``forward_streaming`` and does
        not touch the training-time ``forward()``.
        """
        self = cls.__new__(cls)
        self.device = (
            torch.device(device) if device is not None
            else next(model.parameters()).device
        )
        self.chunk_ms = chunk_ms
        self._model = model
        self._model_cfg = model_cfg
        self.sample_rate = sample_rate

        frames_per_sec = sample_rate // model_cfg.mel.hop_length // 2
        if max_cache_sec is not None:
            model.config.conformer.max_cache_len = int(
                max_cache_sec * frames_per_sec
            )

        if blank_trim_sec is not None and blank_trim_sec > 0:
            self._blank_trim_frames = int(blank_trim_sec * frames_per_sec)
        else:
            self._blank_trim_frames = 0

        self._chunk_samples = int(chunk_ms * sample_rate / 1000)
        self.reset()
        return self

    def reset(self) -> None:
        """Reset all state for a new decoding session."""
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._model_state = self._model.init_streaming_state(self.device)
        self._all_log_probs: list[Tensor] = []
        self._emitted_text = ""
        # Silence-reset bookkeeping. ``_total_frames`` is monotonic across
        # silence-triggered resets so the timer keeps ticking. ``_last_emit_frame``
        # is None until the first character is emitted (startup grace: never
        # reset before the model has produced anything, otherwise initial
        # KV warmup gets nuked the moment the trailing-silence threshold ticks
        # over).
        self._total_frames = 0
        self._last_emit_frame: Optional[int] = None

    def feed_audio(self, audio_chunk: np.ndarray) -> str:
        """Feed raw audio samples, return NEW characters decoded since last call.

        Accumulates audio until a full chunk is ready, then processes
        through the model. Returns empty string if chunk not yet complete.

        Args:
            audio_chunk: 1-D float32 array at self.sample_rate.

        Returns:
            Newly decoded characters (may be empty).
        """
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        new_text = ""
        while len(self._audio_buffer) >= self._chunk_samples:
            chunk = self._audio_buffer[:self._chunk_samples]
            self._audio_buffer = self._audio_buffer[self._chunk_samples:]
            new_text += self._process_chunk(chunk)

        return new_text

    def get_full_text(self) -> str:
        """Get all decoded text so far."""
        if not self._all_log_probs:
            return ""
        all_lp = torch.cat(self._all_log_probs, dim=0)
        return self._greedy_decode(all_lp)

    def flush(self) -> str:
        """Process any remaining buffered audio. Call at end of stream.

        Right-pads the final chunk with ``n_fft // 2`` zeros before
        passing it to the model so the training-time
        ``forward()``-vs-streaming tail frame count matches exactly.
        Training pads both sides of the full utterance with
        ``n_fft // 2`` (see ``MelFrontend.forward``); streaming only
        left-pads the very first chunk. Without this right-pad, the
        final 1-2 mel frames that training saw are missing from the
        streaming output, truncating the tail of the CTC decode.

        Returns any newly decoded characters from the partial chunk.
        """
        if len(self._audio_buffer) > 0:
            n_fft = self._model_cfg.mel.n_fft
            pad_right = n_fft // 2
            chunk = np.concatenate(
                [self._audio_buffer,
                 np.zeros(pad_right, dtype=np.float32)]
            )
            self._audio_buffer = np.zeros(0, dtype=np.float32)
            return self._process_chunk(chunk)
        return ""

    def decode_file(self, path: str) -> str:
        """Decode complete audio file by feeding as streaming chunks."""
        audio = _load_audio(path, self.sample_rate)
        return self.decode_audio(audio)

    def decode_audio(self, audio: np.ndarray) -> str:
        """Decode complete audio array by feeding as streaming chunks.

        Peak-normalizes to the training target amplitude range so the
        log-mel feature distribution at the subsampling input matches
        what the model saw during training (morse_generator normalizes
        every training sample to peak = target_amplitude ∈ [0.5, 0.9]).
        Without this, real recordings with arbitrary peak levels land
        outside the training distribution at block 0.
        """
        self.reset()
        audio = _peak_normalize(audio, target_peak=0.7)
        pos = 0
        while pos < len(audio):
            end = min(pos + self._chunk_samples, len(audio))
            self.feed_audio(audio[pos:end])
            pos = end
        self.flush()
        return self.get_full_text()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Process a single audio chunk through the model.

        Returns newly decoded characters.
        """
        audio_t = torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Compute mel with streaming buffer
            mel, new_stft_buf = self._model.mel_frontend.compute_streaming(
                audio_t, self._model_state.get('stft_buffer'),
            )
            self._model_state['stft_buffer'] = new_stft_buf

            # Forward through model
            log_probs, self._model_state = self._model.forward_streaming(
                mel, self._model_state,
            )

        if log_probs.shape[0] == 0:
            return ""

        # Accumulate log_probs (T, B, C) -> take batch 0 -> (T, C)
        lp = log_probs[:, 0, :].cpu()
        self._all_log_probs.append(lp)
        self._total_frames += lp.shape[0]

        # Decode everything and emit new characters. Log-probs accumulate
        # across silence-triggered resets — only model state is reset, not
        # the decoded sequence — so redecoding the full accumulated log_probs
        # each chunk gives identical text up to the new suffix.
        all_lp = torch.cat(self._all_log_probs, dim=0)
        full_text = self._greedy_decode(all_lp)

        new_chars = full_text[len(self._emitted_text):]
        self._emitted_text = full_text
        if new_chars:
            self._last_emit_frame = self._total_frames

        self._maybe_silence_reset()
        return new_chars

    def _maybe_silence_reset(self) -> None:
        """Flush encoder state if no character has been emitted recently.

        Fires when the elapsed time since the last *committed* CTC emission
        exceeds ``blank_trim_sec``. Uses emission gaps instead of per-frame
        argmax: a real letter occupies only a few of the ~250 trailing frames,
        so per-frame argmax is mostly blank even mid-decode and was firing
        spuriously during initial KV warmup. Mel STFT overlap is preserved so
        the spectrogram stays continuous across the reset; emitted text is
        kept (CTC prefix stability holds across resets).

        Startup grace: never resets before the first character has been
        emitted, otherwise the empty-cache warmup window trips the timer.
        After a fire, the timer restarts at the current frame so the trim
        cannot re-fire until ``blank_trim_sec`` more elapses without an
        emission.
        """
        n = self._blank_trim_frames
        if n <= 0 or self._last_emit_frame is None:
            return
        if (self._total_frames - self._last_emit_frame) < n:
            return

        # Reset model state only. The accumulated log-probs and emitted
        # text are left alone — CTC greedy decode is per-frame and
        # state-independent, so prior frames stay correctly decoded. The
        # post-reset chunks will be computed from fresh KV/conv buffers
        # (good — that's the point), but their per-frame log-probs concat
        # cleanly with the prior segment.
        stft_buffer = self._model_state.get('stft_buffer')
        self._model_state = self._model.init_streaming_state(self.device)
        self._model_state['stft_buffer'] = stft_buffer
        self._last_emit_frame = self._total_frames

    @staticmethod
    def _greedy_decode(log_probs: Tensor) -> str:
        """Greedy CTC decode."""
        if log_probs.shape[0] == 0:
            return ""
        return vocab_module.decode_ctc(log_probs, strip_trailing_space=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode Morse code audio with CW-Former (causal streaming)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to CW-Former checkpoint")
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("--chunk-ms", type=int, default=500, metavar="MS",
                        dest="chunk_ms",
                        help="Processing chunk size in milliseconds")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--blank-trim-sec", type=float, default=5.0,
                        metavar="SEC", dest="blank_trim_sec",
                        help="Reset KV cache after this many seconds of "
                             "blank/space-only output. 0 disables.")

    args = parser.parse_args()

    dec = CWFormerStreamingDecoder(
        checkpoint=args.checkpoint,
        chunk_ms=args.chunk_ms,
        device=args.device,
        blank_trim_sec=args.blank_trim_sec if args.blank_trim_sec > 0 else None,
    )

    print(f"[cwformer-streaming] chunk={dec.chunk_ms}ms "
          f"params={dec._model.num_params:,}")

    transcript = dec.decode_file(args.input)
    print(transcript)


if __name__ == "__main__":
    main()
