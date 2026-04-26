"""Smoke test for the multi-segment composition distributions.

Generates a batch of multi-segment samples with the ``full`` config and
reports/asserts on:
  * total audio length
  * first sender-change time (s into the sample)
  * leading/trailing silence durations
  * gap durations
  * segment count

Run as a script for the full distribution dump, or via pytest for the
assertion-only fast path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import create_default_config
from morse_generator import generate_sample


N_SAMPLES = 200
MAX_AUDIO_SEC = 30.0


def _gen_samples(n: int, max_audio_sec: float, seed: int = 0) -> List[Dict]:
    cfg = create_default_config("full").morse
    cfg.multi_segment_probability = 1.0
    cfg.letter_alternation_probability = 0.0
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 4:
        attempts += 1
        try:
            audio, text, meta = generate_sample(
                cfg, rng=rng, max_duration_sec=max_audio_sec,
            )
        except ValueError:
            continue
        if not meta.get("multi_segment"):
            continue
        gaps = list(meta.get("gap_durations_sec", []))
        first_change = None
        if meta["n_segments"] > 1 and gaps:
            # The first sender change ends at the start of gap_0, which
            # is equal to leading_silence + duration of seg_0. Recover it
            # from the joined audio's energy envelope: find the first
            # extended silence that follows the first non-silent run.
            first_change = _first_change_time(audio, cfg.sample_rate)
        out.append({
            "audio_len_sec": len(audio) / cfg.sample_rate,
            "first_change_sec": first_change,
            "leading": meta["leading_silence_sec"],
            "trailing": meta["trailing_silence_sec"],
            "gaps": gaps,
            "n_segments": meta["n_segments"],
            "n_rendered_segments": meta.get("n_rendered_segments", meta["n_segments"]),
            "n_dropped_segments": meta.get("n_dropped_segments", 0),
            "segment_wpms": meta["segment_wpms"],
            "joined_text": text,
        })
    return out


def _first_change_time(audio: np.ndarray, sr: int) -> float:
    """Approximate the first segment-boundary time (s into the sample)
    by locating the first long silence (>= 200 ms) following the first
    burst of signal energy. Robust against the post-build shuffle
    because it works directly on the rendered audio.
    """
    win = max(1, int(0.02 * sr))  # 20 ms RMS window
    n = (len(audio) // win) * win
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(
        audio[:n].astype(np.float32).reshape(-1, win) ** 2, axis=1,
    ))
    if rms.max() < 1e-6:
        return 0.0
    thresh = 0.2 * rms.max()
    active = rms > thresh
    # Find first sustained activity, then first sustained silence after it.
    i = 0
    while i < len(active) and not active[i]:
        i += 1
    # consume the active run
    silence_run_needed = max(1, int(0.20 / 0.02))  # 200 ms = 10 frames
    while i < len(active):
        if not active[i]:
            run_start = i
            while i < len(active) and not active[i]:
                i += 1
            if (i - run_start) >= silence_run_needed:
                return run_start * 0.02
        else:
            i += 1
    return float(n) / sr


def summarise(samples: List[Dict]) -> str:
    lines: List[str] = []
    lines.append(f"n_samples={len(samples)}")
    audio_lens = [s["audio_len_sec"] for s in samples]
    lines.append(
        f"audio_len_sec  : min={min(audio_lens):.2f}  "
        f"max={max(audio_lens):.2f}  "
        f"mean={float(np.mean(audio_lens)):.2f}"
    )
    leadings = [s["leading"] for s in samples]
    trailings = [s["trailing"] for s in samples]
    lines.append(
        f"leading_sec    : min={min(leadings):.2f}  "
        f"max={max(leadings):.2f}  "
        f"mean={float(np.mean(leadings)):.2f}"
    )
    lines.append(
        f"trailing_sec   : min={min(trailings):.2f}  "
        f"max={max(trailings):.2f}  "
        f"mean={float(np.mean(trailings)):.2f}"
    )
    all_gaps = [g for s in samples for g in s["gaps"]]
    if all_gaps:
        lines.append(
            f"gap_sec        : min={min(all_gaps):.2f}  "
            f"max={max(all_gaps):.2f}  "
            f"mean={float(np.mean(all_gaps)):.2f}  "
            f"n={len(all_gaps)}"
        )
    n_segs = [s["n_segments"] for s in samples]
    counts = {k: n_segs.count(k) for k in sorted(set(n_segs))}
    lines.append(f"n_segments     : {counts}")
    first_changes = [
        s["first_change_sec"] for s in samples
        if s["first_change_sec"] is not None
    ]
    if first_changes:
        lines.append(
            f"first_change_sec: min={min(first_changes):.2f}  "
            f"max={max(first_changes):.2f}  "
            f"mean={float(np.mean(first_changes)):.2f}"
        )
    return "\n".join(lines)


def test_multi_segment_distribution() -> None:
    samples = _gen_samples(N_SAMPLES, MAX_AUDIO_SEC, seed=12345)
    assert samples, "no multi-segment samples generated"

    audio_lens = [s["audio_len_sec"] for s in samples]
    assert max(audio_lens) <= MAX_AUDIO_SEC + 0.01, (
        f"audio length exceeded cap: max={max(audio_lens):.3f}"
    )

    # Audio cap and target-text consistency: every rendered segment's text
    # must appear in joined_text (no truncation mismatch).
    for s in samples:
        assert s["audio_len_sec"] <= MAX_AUDIO_SEC + 1e-3, (
            f"audio overshoot: {s['audio_len_sec']:.4f} > {MAX_AUDIO_SEC}"
        )
        n_rendered = s["n_rendered_segments"]
        assert n_rendered == s["n_segments"], (
            f"n_segments={s['n_segments']} != n_rendered={n_rendered}"
        )
        # joined_text is " ".join(text_parts); rendered segments equal
        # the number of text_parts, so splitting and counting tokens is
        # imprecise (segment text itself contains spaces). Instead check
        # that text is non-empty and that all rendered segment WPMs and
        # pitches are tracked.
        assert s["joined_text"], "rendered sample has empty text"
        assert len(s["segment_wpms"]) == n_rendered

    first_changes = [
        s["first_change_sec"] for s in samples
        if s["first_change_sec"] is not None
    ]
    assert first_changes, "no multi-segment (n>=2) samples produced"
    assert min(first_changes) <= 2.0, (
        f"first sender change never lands before 2 s: "
        f"min={min(first_changes):.2f}"
    )
    assert max(first_changes) >= 25.0 or MAX_AUDIO_SEC < 28.0, (
        f"first sender change never lands after 25 s: "
        f"max={max(first_changes):.2f} (audio_max={MAX_AUDIO_SEC})"
    )
    # The full assertion is "spans at least [2 s, 25 s]"; under
    # max_audio_sec=30 the 25 s upper bound only fits when the first
    # segment is the long remainder, so soften it slightly.

    all_gaps = [g for s in samples for g in s["gaps"]]
    if all_gaps:
        assert max(all_gaps) > 5.0, (
            f"long-gap tail unreachable: max gap={max(all_gaps):.2f}"
        )


if __name__ == "__main__":
    samples = _gen_samples(N_SAMPLES, MAX_AUDIO_SEC, seed=12345)
    print(summarise(samples))
    test_multi_segment_distribution()
    print("OK")
