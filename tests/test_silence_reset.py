"""
Silence-triggered KV-cache reset test for CWFormerStreamingDecoder.

Feeds: 2 s of synthetic Morse-ish tone, then silence in two stages, then
more tone. We check the streaming state right after the silence-trigger
fires (before subsequent silent chunks re-accumulate pos_offset). At that
moment ``pos_offset`` must be 0 and every per-layer KV cache must be
empty. We also verify that ``blank_trim_sec=0`` disables the behavior.

Skipped if no checkpoint is present at ``checkpoints_cwformer/best_model.pt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


CHECKPOINT_PATH = Path("checkpoints_cwformer/best_model.pt")


def _make_dit_tone(sample_rate: int, duration_sec: float, freq: float = 600.0,
                   wpm: int = 20) -> np.ndarray:
    """Synthesise a gated 600 Hz tone alternating between dit and silence.

    Not a faithful Morse signal — just enough non-silence frames to drive
    the model off the blank/space classes so the silence-reset arming
    flag flips on.
    """
    dit_sec = 1.2 / wpm
    n_total = int(duration_sec * sample_rate)
    t = np.arange(n_total, dtype=np.float32) / sample_rate
    tone = 0.7 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    period = int(2 * dit_sec * sample_rate)
    on = int(dit_sec * sample_rate)
    gate = np.zeros(n_total, dtype=np.float32)
    for start in range(0, n_total, period):
        gate[start:start + on] = 1.0
    return tone * gate


def _kv_lengths(state: dict) -> list[int]:
    return [k.shape[2] for (k, _v) in state["kv_caches"]]


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"No checkpoint at {CHECKPOINT_PATH}",
)
def test_silence_resets_kv_cache():
    from neural_decoder.inference_cwformer import CWFormerStreamingDecoder

    dec = CWFormerStreamingDecoder(
        checkpoint=str(CHECKPOINT_PATH),
        chunk_ms=500,
        device="cpu",
        blank_trim_sec=5.0,
    )
    sr = dec.sample_rate

    tone1 = _make_dit_tone(sr, 2.0)
    tone2 = _make_dit_tone(sr, 2.0)
    chunk_samples = int(0.5 * sr)

    dec.reset()
    dec.feed_audio(tone1)
    assert dec._model_state["pos_offset"] > 0
    assert max(_kv_lengths(dec._model_state)) > 0
    pos_after_tone = dec._model_state["pos_offset"]

    fired_at_chunk = -1
    for i in range(20):
        dec.feed_audio(np.zeros(chunk_samples, dtype=np.float32))
        if dec._model_state["pos_offset"] < pos_after_tone:
            fired_at_chunk = i
            break

    assert fired_at_chunk >= 0, (
        "silence-triggered reset never fired within 10 s of silence"
    )
    assert dec._model_state["pos_offset"] == 0, (
        f"pos_offset should be 0 right after reset, "
        f"got {dec._model_state['pos_offset']}"
    )
    assert all(L == 0 for L in _kv_lengths(dec._model_state)), (
        f"all KV caches should be empty after reset, "
        f"got {_kv_lengths(dec._model_state)}"
    )
    assert dec._model_state["stft_buffer"] is not None, (
        "stft_buffer must be preserved across silence-triggered reset"
    )

    dec.feed_audio(tone2)
    assert dec._model_state["pos_offset"] > 0, (
        "pos_offset should grow again after non-silent audio post-reset"
    )


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"No checkpoint at {CHECKPOINT_PATH}",
)
def test_blank_trim_sec_zero_disables():
    from neural_decoder.inference_cwformer import CWFormerStreamingDecoder

    dec = CWFormerStreamingDecoder(
        checkpoint=str(CHECKPOINT_PATH),
        chunk_ms=500,
        device="cpu",
        blank_trim_sec=0,
    )
    sr = dec.sample_rate

    tone = _make_dit_tone(sr, 2.0)

    dec.reset()
    dec.feed_audio(tone)
    pos_after_tone = dec._model_state["pos_offset"]
    assert pos_after_tone > 0

    chunk_samples = int(0.5 * sr)
    for _ in range(14):
        dec.feed_audio(np.zeros(chunk_samples, dtype=np.float32))
    assert dec._model_state["pos_offset"] > pos_after_tone, (
        "blank_trim_sec=0 must NOT reset state; pos_offset should keep growing."
    )
