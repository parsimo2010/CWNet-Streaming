#!/usr/bin/env python3
"""
diag_realworld.py -- Per-frame model diagnostic on real-world recordings.

For each WAV file matched by --pattern (default: ``p3hosting*.wav``) under
--recordings, run streaming ONNX decode, capture per-CTC-frame log_probs,
and produce a diagnostic plot stacked on a shared time axis:

    1. Mel spectrogram (the actual model input)
    2. Decoded character labels with vertical alignment lines
    3. Filtered-audio envelope (smoothed |audio|) at mel rate
    4. Per-frame probabilities: P(blank), P(space), max P(letter)
    5. Per-frame entropy of the softmax distribution

Suspicious windows -- where the audio envelope is clearly above the
ambient floor but the model emits high-confidence blanks for >= 0.5 s --
are highlighted in red across all panels and listed in the summary.

Use this to distinguish:
  - High P(blank) + low entropy   --> model is confident it's silence
  - High entropy, no clear winner --> model is uncertain (OOD-like)
  - max P(letter) just below P(blank) --> a near-miss commit

Outputs:
  <output>/<basename>_diag.png    one figure per recording
  <output>/summary.txt            transcript + stats per file

Usage::

    python tests/diagnostic/diag_realworld.py
    python tests/diagnostic/diag_realworld.py \
        --recordings recordings --pattern 'p3hosting*.wav' \
        --model deploy/cwformer_streaming_fp32.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sibling modules importable.
_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY = _ROOT / "deploy"
for p in (_DEPLOY, _ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference_onnx import (  # type: ignore  # noqa: E402
    BLANK_IDX,
    IDX_TO_CHAR,
    CWFormerStreamingONNX,
    MelComputer,
    _peak_normalize,
    load_audio,
)


SPACE_IDX = 1


def greedy_with_frames(log_probs: np.ndarray) -> List[Tuple[str, int]]:
    if log_probs.size == 0:
        return []
    indices = np.argmax(log_probs, axis=-1)
    out: List[Tuple[str, int]] = []
    prev = -1
    for t, idx in enumerate(indices):
        idx_i = int(idx)
        if idx_i != prev:
            if idx_i != BLANK_IDX:
                ch = IDX_TO_CHAR.get(idx_i, "")
                if ch:
                    out.append((ch, t))
            prev = idx_i
    return out


def run_streaming(
    dec: CWFormerStreamingONNX, audio: np.ndarray,
) -> np.ndarray:
    """Run streaming decode end-to-end; return concatenated log_probs (T, C)."""
    dec.reset()
    pos = 0
    chunk = dec._chunk_samples
    while pos < len(audio):
        end = min(pos + chunk, len(audio))
        c = audio[pos:end]
        if len(c) < chunk:
            c = np.concatenate(
                [c, np.zeros(chunk - len(c), dtype=np.float32)])
        dec.feed_audio(c)
        pos = end
    dec.flush()
    if not dec._all_log_probs:
        return np.zeros((0, len(IDX_TO_CHAR) + 1), dtype=np.float32)
    return np.concatenate(dec._all_log_probs, axis=0)


# ---------------------------------------------------------------------------
# Detection / pitch helpers
# ---------------------------------------------------------------------------

def envelope_swing(env: np.ndarray, window: int) -> np.ndarray:
    """Sliding (max - min) over `window` samples, padded to original length.

    A bandpass-filtered noise envelope (even with AGC) has a relatively
    constant level, so its swing is small. CW keying produces alternation
    between near-zero (space) and high (mark), so swing is large. This
    is the right discriminator for "code present" — absolute level is
    not, since AGC normalises it.
    """
    n = len(env)
    if n < window or window < 2:
        return np.zeros_like(env)
    shape = (n - window + 1, window)
    strides = (env.strides[0], env.strides[0])
    view = np.lib.stride_tricks.as_strided(env, shape=shape, strides=strides)
    swing = (view.max(axis=1) - view.min(axis=1)).astype(env.dtype)
    pad_l = window // 2
    pad_r = n - len(swing) - pad_l
    pad_l_arr = np.full(pad_l, swing[0], dtype=env.dtype)
    pad_r_arr = np.full(max(pad_r, 0), swing[-1], dtype=env.dtype)
    return np.concatenate([pad_l_arr, swing, pad_r_arr])[:n]


def compute_pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    hop_sec: float = 0.01,
    win_sec: float = 0.05,
    fmin: float = 100.0,
    fmax: float = 2500.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window dominant frequency (Hz) inside [fmin, fmax].

    Returns ``(times, pitches, powers)``. ``pitches`` is NaN where the
    window's peak power is below the file-wide noise floor (no clear
    spectral peak).
    """
    hop = int(hop_sec * sample_rate)
    win = int(win_sec * sample_rate)
    if len(audio) < win:
        return (np.zeros(0), np.zeros(0), np.zeros(0))
    n_windows = (len(audio) - win) // hop + 1
    times = np.arange(n_windows) * hop_sec
    pitches = np.full(n_windows, np.nan, dtype=np.float32)
    powers = np.zeros(n_windows, dtype=np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = freqs[band_mask]
    han = np.hanning(win).astype(np.float32)
    for i in range(n_windows):
        seg = audio[i * hop:i * hop + win] * han
        spec = np.abs(np.fft.rfft(seg))[band_mask]
        powers[i] = float(spec.max())
        pitches[i] = float(band_freqs[spec.argmax()])
    # Mask out low-power windows where the "peak" is just noise.
    if powers.size:
        floor = max(np.percentile(powers, 25) * 1.5, 1e-3)
        pitches[powers < floor] = np.nan
    return times, pitches, powers


def find_transcript_gaps(
    chars: List[Tuple[str, int]],
    duration: float,
    min_gap_sec: float = 1.0,
) -> List[Tuple[float, float]]:
    """Time intervals where the streamed decoder produced no letters.

    Each char's time is its CTC frame index * 0.02 s. A gap is any
    interval >= ``min_gap_sec`` between consecutive emitted chars
    (or from the start, or to the end of audio).
    """
    times = [j * 0.02 for _, j in chars]
    gaps: List[Tuple[float, float]] = []
    prev = 0.0
    for t in times:
        if t - prev >= min_gap_sec:
            gaps.append((prev, t))
        prev = t
    if duration - prev >= min_gap_sec:
        gaps.append((prev, duration))
    return gaps


def isolated_redecode(
    dec: CWFormerStreamingONNX,
    audio: np.ndarray,
    t0: float,
    t1: float,
    pad_sec: float = 0.0,
    renormalize: bool = False,
) -> str:
    """Run a fresh-state decode on the audio slice [t0-pad, t1+pad].

    With ``renormalize=False`` (default) the audio is fed at its
    original amplitude. ``audio`` is assumed to be already peak-
    normalized at the file level (same as the streamed decode), so
    this gives an apples-to-apples state-only comparison: the
    streamed decoder runs THIS segment with 50 s of accumulated KV/
    conv state, the isolated decoder runs the SAME segment at the
    SAME amplitude with zero state. Difference => state drift.

    With ``renormalize=True`` we also peak-normalize the segment to
    0.7 — useful for asking "would amplitude OOD explain this?".
    """
    sr = dec.sample_rate
    a = max(0.0, t0 - pad_sec)
    b = min(len(audio) / sr, t1 + pad_sec)
    seg = audio[int(a * sr):int(b * sr)].astype(np.float32)
    if seg.size == 0:
        return ""
    if renormalize:
        return dec.decode_audio(seg)
    # Manual: reset, feed_audio (no normalize), flush, read text.
    dec.reset()
    pos = 0
    chunk = dec._chunk_samples
    while pos < len(seg):
        end = min(pos + chunk, len(seg))
        c = seg[pos:end]
        if len(c) < chunk:
            c = np.concatenate(
                [c, np.zeros(chunk - len(c), dtype=np.float32)])
        dec.feed_audio(c)
        pos = end
    dec.flush()
    return dec.get_full_text()


def diagnose_one(
    path: Path,
    dec: CWFormerStreamingONNX,
    mel_disp_computer: MelComputer,
    output_dir: Path,
) -> Dict:
    audio = load_audio(str(path), dec.sample_rate)
    audio = _peak_normalize(audio, target_peak=0.7)

    log_probs = run_streaming(dec, audio)
    T_ctc = log_probs.shape[0]

    if T_ctc == 0:
        return {"path": str(path), "skipped": "no log_probs"}

    # Probabilities + entropy
    probs = np.exp(log_probs).astype(np.float64)
    p_blank = probs[:, BLANK_IDX]
    p_space = probs[:, SPACE_IDX]
    p_letter_max = probs[:, 2:].max(axis=-1)
    # Entropy in nats. Use clip to avoid 0 * -inf.
    p_clip = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(p_clip), axis=-1)

    # Audio envelope at mel rate (10 ms)
    win = 160
    env = np.abs(audio)
    env_smooth = np.convolve(env, np.ones(win) / win, mode="same")
    env_mel = env_smooth[::160].astype(np.float32)

    # Envelope swing (max - min over a 300 ms sliding window at mel
    # rate). AGC keeps the noise envelope near-constant level (low
    # swing) but cannot suppress the binary on/off modulation of CW
    # keying (high swing). Absolute level is uninformative under AGC.
    swing = envelope_swing(env_mel, window=30)

    # Pitch track over the full audio.
    pitch_t, pitch_hz, _ = compute_pitch_track(
        audio, dec.sample_rate, hop_sec=0.01, win_sec=0.05,
        fmin=100.0, fmax=2500.0,
    )

    # Mel for display.
    mel_full, n_mel_frames = mel_disp_computer.compute(audio)
    mel_disp = mel_full[0].T

    chars = greedy_with_frames(log_probs)
    transcript = "".join(ch for ch, _ in chars).strip()
    duration = len(audio) / dec.sample_rate

    # === Gap-based suspicious-window detection ===
    # A "gap" is any interval >= 1 s between successive emitted letters.
    # We classify each gap by mean envelope swing AND by what a
    # fresh-state isolated decode of the SAME interval (no padding)
    # produces. This separates three regimes:
    #
    #   high swing + isolated finds letters --> state drift in the
    #     streamed run; the model can decode the audio with fresh state.
    #   high swing + isolated also empty   --> input-level OOD;
    #     real audio property the model can't handle.
    #   low swing                           --> genuine inter-transmit
    #     silence, no recovery expected.
    gaps = find_transcript_gaps(chars, duration, min_gap_sec=1.0)
    gap_results: List[Dict] = []
    for (t0, t1) in gaps:
        i0 = int(t0 * 100)
        i1 = int(t1 * 100)
        seg_swing = swing[i0:i1] if i1 > i0 else swing[i0:i0 + 1]
        mean_sw = float(np.mean(seg_swing)) if seg_swing.size else 0.0
        # Two isolated re-decodes:
        #   iso_same_gain : same amplitude as the streamed run
        #                   --> isolates the effect of fresh state.
        #   iso_renorm    : peak-normalized to 0.7
        #                   --> shows what the model can do if also
        #                       given an amplitude boost.
        try:
            iso_same = isolated_redecode(
                dec, audio, t0, t1, pad_sec=0.0, renormalize=False,
            )
        except Exception as e:
            iso_same = f"<error: {type(e).__name__}>"
        try:
            iso_renorm = isolated_redecode(
                dec, audio, t0, t1, pad_sec=0.0, renormalize=True,
            )
        except Exception as e:
            iso_renorm = f"<error: {type(e).__name__}>"
        gap_results.append({
            "t0": t0, "t1": t1,
            "duration": t1 - t0,
            "mean_swing": mean_sw,
            "isolated_same_gain": iso_same,
            "isolated_renorm": iso_renorm,
        })

    SWING_CODE_THRESH = 0.10
    # "Suspicious" = gap with significant swing AND fresh-state
    # isolated decode (at SAME gain) recovers letters the streamed
    # decoder missed. That's the state-drift case.
    suspicious_windows = [
        (g["t0"], g["t1"]) for g in gap_results
        if g["mean_swing"] > SWING_CODE_THRESH
        and g["isolated_same_gain"].strip()
    ]

    t_mel = np.arange(n_mel_frames) * 0.01
    t_ctc = np.arange(T_ctc) * 0.02
    t_env = np.arange(len(env_mel)) * 0.01
    cfg = dec.config

    # === Plot ===
    fig_w = max(16.0, min(60.0, duration * 0.4))
    fig = plt.figure(figsize=(fig_w, 14), dpi=100)
    gs = fig.add_gridspec(
        6, 1,
        height_ratios=[3.5, 0.6, 1.5, 0.9, 1.5, 1.0],
        hspace=0.10, left=0.05, right=0.99, top=0.95, bottom=0.06,
    )
    ax_mel = fig.add_subplot(gs[0])
    ax_letters = fig.add_subplot(gs[1], sharex=ax_mel)
    ax_env = fig.add_subplot(gs[2], sharex=ax_mel)
    ax_pitch = fig.add_subplot(gs[3], sharex=ax_mel)
    ax_probs = fig.add_subplot(gs[4], sharex=ax_mel)
    ax_ent = fig.add_subplot(gs[5], sharex=ax_mel)

    ax_mel.imshow(
        mel_disp, aspect="auto", origin="lower", cmap="viridis",
        extent=[0.0, n_mel_frames * 0.01, cfg["f_min"], cfg["f_max"]],
        vmin=-10.0, vmax=2.0, interpolation="nearest",
    )
    ax_mel.set_ylabel("Mel (Hz)")
    short_tx = transcript if len(transcript) <= 200 else transcript[:200] + "..."
    ax_mel.set_title(
        f'{path.name}   ({duration:.1f}s)   decoded: "{short_tx}"',
        fontsize=10, loc="left",
    )
    ax_mel.tick_params(labelbottom=False)

    # Letters
    ax_letters.set_ylim(0, 1)
    ax_letters.set_yticks([])
    for sp in ("top", "right", "left"):
        ax_letters.spines[sp].set_visible(False)
    ax_letters.tick_params(labelbottom=False, bottom=False)
    for k, (ch, j) in enumerate(chars):
        t = j * 0.02
        y = 0.65 if k % 2 == 0 else 0.30
        ax_letters.text(
            t, y, ch, ha="center", va="center",
            fontsize=8, family="monospace", color="black",
        )
        ax_letters.axvline(t, color="black", alpha=0.15, linewidth=0.4)

    # Envelope + swing curve (the code-presence indicator)
    ax_env.plot(t_env, env_mel, lw=0.5, color="black", label="|audio|")
    ax_env.plot(
        t_env[: len(swing)], swing, lw=0.7, color="orange",
        label="swing (max-min, 300ms)",
    )
    ax_env.axhline(
        SWING_CODE_THRESH, color="red", lw=0.5, ls="--", alpha=0.7,
        label=f"code thr={SWING_CODE_THRESH:.2f}",
    )
    ax_env.set_ylabel("env / swing")
    ax_env.legend(loc="upper right", fontsize=7, ncol=3)
    ax_env.tick_params(labelbottom=False)

    # Pitch track. Shaded area = mel-band coverage.
    ax_pitch.axhspan(
        cfg["f_min"], cfg["f_max"], color="gray", alpha=0.12,
        label=f"mel band {int(cfg['f_min'])}-{int(cfg['f_max'])} Hz",
    )
    ax_pitch.plot(pitch_t, pitch_hz, ".", ms=1.0, color="darkgreen")
    ax_pitch.set_ylabel("Pitch (Hz)")
    ax_pitch.set_ylim(100, 2500)
    ax_pitch.legend(loc="upper right", fontsize=7)
    ax_pitch.tick_params(labelbottom=False)

    # Probabilities
    ax_probs.plot(t_ctc, p_blank, label="P(blank)", color="gray", lw=0.5)
    ax_probs.plot(t_ctc, p_space, label="P(space)", color="blue", lw=0.5)
    ax_probs.plot(
        t_ctc, p_letter_max, label="max P(letter)", color="red", lw=0.5,
    )
    ax_probs.set_ylim(-0.02, 1.05)
    ax_probs.set_ylabel("Probability")
    ax_probs.legend(loc="upper right", fontsize=7)
    ax_probs.tick_params(labelbottom=False)

    # Entropy
    ax_ent.plot(t_ctc, entropy, lw=0.5, color="purple")
    ax_ent.set_ylabel("Entropy (nats)")
    ax_ent.set_xlabel("Time (s)")

    # Suspicious shading on all axes
    for (t0, t1) in suspicious_windows:
        for ax in (ax_mel, ax_letters, ax_env, ax_pitch, ax_probs, ax_ent):
            ax.axvspan(t0, t1, color="red", alpha=0.10, zorder=-10)

    out_path = output_dir / f"{path.stem}_diag.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    # Pitch-band excursion: fraction of code-active frames (swing >
    # threshold) whose dominant pitch falls outside the mel band.
    pitch_excursion = 0.0
    if pitch_hz.size and swing.size:
        n = min(len(pitch_hz), len(swing))
        active = swing[:n] > SWING_CODE_THRESH
        ph = pitch_hz[:n]
        if active.sum() > 0:
            outside = (
                ((ph < cfg["f_min"]) | (ph > cfg["f_max"]))
                & active & ~np.isnan(ph)
            )
            pitch_excursion = float(outside.sum() / active.sum())

    return {
        "path": str(path),
        "duration_s": float(duration),
        "n_chars_decoded": len(chars),
        "transcript": transcript,
        "mean_p_blank": float(p_blank.mean()),
        "median_entropy": float(np.median(entropy)),
        "max_entropy": float(entropy.max()),
        "pitch_outside_band_frac": pitch_excursion,
        "gap_results": gap_results,
        "suspicious_windows_s": suspicious_windows,
        "longest_suspicious_s": (
            max((t1 - t0) for t0, t1 in suspicious_windows)
            if suspicious_windows else 0.0
        ),
        "total_suspicious_s": float(
            sum((t1 - t0) for t0, t1 in suspicious_windows)
        ),
        "png": str(out_path),
    }


def write_summary(results: List[Dict], output_dir: Path) -> None:
    lines = []
    lines.append(f"# CWformer real-world diagnostic summary")
    lines.append(f"# {len(results)} file(s)")
    lines.append("")
    for r in results:
        if "skipped" in r:
            lines.append(f"## {Path(r['path']).name} -- SKIPPED ({r['skipped']})")
            lines.append("")
            continue
        lines.append(f"## {Path(r['path']).name}")
        lines.append(f"  duration:                {r['duration_s']:.1f} s")
        lines.append(f"  chars decoded:           {r['n_chars_decoded']}")
        lines.append(f"  mean P(blank):           {r['mean_p_blank']:.3f}")
        lines.append(f"  median entropy:          {r['median_entropy']:.3f} nats")
        lines.append(f"  max entropy:             {r['max_entropy']:.3f} nats")
        lines.append(
            f"  pitch outside mel band:  "
            f"{r['pitch_outside_band_frac'] * 100:.1f}% of code-active frames"
        )
        lines.append(
            f"  suspicious windows:      {len(r['suspicious_windows_s'])} "
            f"(total {r['total_suspicious_s']:.1f} s, "
            f"longest {r['longest_suspicious_s']:.1f} s)"
        )
        if r["gap_results"]:
            lines.append(
                "  transcript gaps (>= 1.0 s with no streamed letters):"
            )
            lines.append(
                "      time           dur   swing   isolated@same-gain  "
                "| isolated@peak-0.7   classification"
            )
            for g in r["gap_results"]:
                iso_s = g["isolated_same_gain"]
                iso_r = g["isolated_renorm"]
                iso_s_short = iso_s if len(iso_s) <= 28 else iso_s[:28] + "..."
                iso_r_short = iso_r if len(iso_r) <= 28 else iso_r[:28] + "..."
                if g["mean_swing"] <= 0.10:
                    tag = "silence"
                elif iso_s.strip():
                    tag = "STATE-DRIFT"
                elif iso_r.strip():
                    tag = "AMPLITUDE-OOD"
                else:
                    tag = "INPUT-OOD"
                lines.append(
                    f"      [{g['t0']:6.2f}-{g['t1']:6.2f}] "
                    f"{g['duration']:4.1f}s "
                    f"sw={g['mean_swing']:.3f}  "
                    f"{iso_s_short!r:<32}  "
                    f"{iso_r_short!r:<32}  "
                    f"{tag}"
                )
        lines.append(f"  transcript:              {r['transcript']}")
        lines.append(f"  png:                     {Path(r['png']).name}")
        lines.append("")
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default=str(_DEPLOY / "cwformer_streaming_fp32.onnx"),
        help="ONNX model path (FP32 recommended for diagnostics)",
    )
    parser.add_argument(
        "--recordings", default=str(_ROOT / "recordings"),
        help="Directory containing recordings",
    )
    parser.add_argument(
        "--pattern", default="p3hosting*.wav",
        help="Glob pattern within --recordings",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output dir (default: <recordings>/diagnostics)",
    )
    args = parser.parse_args()

    rec_dir = Path(args.recordings)
    out_dir = Path(args.output) if args.output else rec_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rec_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {rec_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[diag] model: {args.model}")
    print(f"[diag] {len(files)} file(s) in {rec_dir} matching {args.pattern}")
    print(f"[diag] output: {out_dir}")

    dec = CWFormerStreamingONNX(model_path=args.model)
    mel_computer = MelComputer(
        dec.config, config_dir=str(Path(args.model).parent),
    )

    results: List[Dict] = []
    for i, path in enumerate(files, 1):
        print(f"[diag] ({i}/{len(files)}) {path.name}", flush=True)
        try:
            r = diagnose_one(path, dec, mel_computer, out_dir)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            r = {"path": str(path), "skipped": str(e)}
        results.append(r)

    write_summary(results, out_dir)


if __name__ == "__main__":
    main()
