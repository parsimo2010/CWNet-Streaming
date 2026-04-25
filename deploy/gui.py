#!/usr/bin/env python3
"""
gui.py -- Cross-platform GUI for streaming CWformer ONNX inference.

Displays:
  - Raw audio waterfall (linear-frequency STFT power, log scale)
  - Mel spectrogram (the actual model input)
  - Decoded text in a scrolling text box
  - Per-character markers on top of the spectrograms, aligned with the
    mel frame where greedy CTC first emitted that character. The marker
    moves with the spectrogram as new audio scrolls in.

Sources:
  - Audio file (WAV / FLAC / OGG / ...)
  - Live audio input device
  - Stdin (raw 16-bit PCM, 16 kHz mono)

Works with the FP32 or INT8 ONNX model exported by quantize_cwformer.py.

Usage::

    python deploy/gui.py
    python deploy/gui.py --model deploy/cwformer_streaming_fp32.onnx

Requires:
  numpy, soundfile, onnxruntime, matplotlib, tkinter (stdlib)
Optional:
  sounddevice (live device input), scipy (resampling fallback)
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Make sibling inference_onnx.py importable when run from any CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from inference_onnx import (  # type: ignore  # noqa: E402
    BLANK_IDX,
    IDX_TO_CHAR,
    CWFormerStreamingONNX,
    MelComputer,
    _peak_normalize,
    device_stream,
    load_audio,
)

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

WINDOW_FRAMES = 1000        # Mel frames visible at once (10 s @ 100 fps).
WATERFALL_MAX_HZ = 3000.0   # Crop linear waterfall to this freq for display.
POLL_INTERVAL_MS = 50       # Tk after() poll cadence.


# ---------------------------------------------------------------------------
# Streaming linear-frequency power spectrogram
# ---------------------------------------------------------------------------

class StreamingPower:
    """Streaming log-power spectrogram (no mel filterbank).

    Mirrors ``MelComputer.compute_streaming`` exactly so its frame grid
    aligns with the mel display and the model input. Returns log power
    per FFT bin.
    """

    def __init__(self, n_fft: int, hop: int) -> None:
        self.n_fft = n_fft
        self.hop = hop
        self.window = (
            0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
        ).astype(np.float32)
        self.n_freqs = n_fft // 2 + 1

    def compute_streaming(
        self,
        audio_chunk: np.ndarray,
        stft_buffer: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if stft_buffer is not None:
            audio = np.concatenate([stft_buffer, audio_chunk]).astype(np.float32)
        else:
            audio = np.pad(audio_chunk, (self.n_fft // 2, 0)).astype(np.float32)

        audio_len = len(audio)
        n_frames = (audio_len - self.n_fft) // self.hop + 1 if audio_len >= self.n_fft else 0

        if n_frames > 0:
            consumed_up_to = n_frames * self.hop
            new_buffer = audio[consumed_up_to:].copy()
        else:
            new_buffer = audio.copy()

        if n_frames <= 0:
            return np.zeros((0, self.n_freqs), dtype=np.float32), new_buffer

        stft_len = (n_frames - 1) * self.hop + self.n_fft
        audio_for_stft = audio[:stft_len]
        shape = (n_frames, self.n_fft)
        strides = (audio_for_stft.strides[0] * self.hop, audio_for_stft.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            audio_for_stft, shape=shape, strides=strides
        )
        windowed = frames * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        power = (np.abs(spec) ** 2).astype(np.float32)
        log_power = np.log(power + 1e-6).astype(np.float32)
        return log_power, new_buffer


# ---------------------------------------------------------------------------
# Frame-tracked greedy CTC decode
# ---------------------------------------------------------------------------

def greedy_with_frames(log_probs: np.ndarray) -> List[Tuple[str, int]]:
    """Greedy CTC decode that returns (char, ctc_frame_idx) tuples.

    The frame index is the FIRST frame of the run that produced the
    character (the moment greedy decode commits it).
    """
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


# ---------------------------------------------------------------------------
# Audio sources (each yields chunk_samples-sized float32 arrays)
# ---------------------------------------------------------------------------

def file_source(
    path: str,
    sample_rate: int,
    chunk_samples: int,
    stop_event: threading.Event,
    realtime: bool = True,
) -> Iterator[np.ndarray]:
    """Yield chunks from a file at real-time pace, peak-normalized to 0.7
    so feature distribution matches training (mirroring decode_audio)."""
    audio = load_audio(path, sample_rate)
    audio = _peak_normalize(audio, target_peak=0.7)
    pos = 0
    chunk_dur = chunk_samples / sample_rate
    t0 = time.monotonic()
    i = 0
    while pos < len(audio) and not stop_event.is_set():
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        if len(chunk) < chunk_samples:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_samples - len(chunk), dtype=np.float32)]
            )
        yield chunk
        pos = end
        i += 1
        if realtime:
            target = t0 + i * chunk_dur
            while not stop_event.is_set():
                remaining = target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))


def stdin_source(
    chunk_samples: int,
    stop_event: threading.Event,
) -> Iterator[np.ndarray]:
    chunk_bytes = chunk_samples * 2  # int16
    while not stop_event.is_set():
        data = sys.stdin.buffer.read(chunk_bytes)
        if not data:
            break
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < chunk_samples:
            audio = np.concatenate(
                [audio, np.zeros(chunk_samples - len(audio), dtype=np.float32)]
            )
        yield audio


def device_source(
    sample_rate: int,
    chunk_samples: int,
    device_id: Optional[int],
    stop_event: threading.Event,
) -> Iterator[np.ndarray]:
    """Re-frame device_stream's chunks into chunk_samples-sized blocks."""
    gen = device_stream(target_sr=sample_rate, device=device_id, chunk_ms=100.0)
    buf = np.zeros(0, dtype=np.float32)
    try:
        for raw in gen:
            if stop_event.is_set():
                break
            buf = np.concatenate([buf, raw])
            while len(buf) >= chunk_samples and not stop_event.is_set():
                yield buf[:chunk_samples].copy()
                buf = buf[chunk_samples:]
    finally:
        gen.close()


# ---------------------------------------------------------------------------
# Tk GUI
# ---------------------------------------------------------------------------

class CWformerGUI:
    def __init__(self, root: tk.Tk, default_model: Optional[str]) -> None:
        self.root = root
        self.root.title("CWformer Decoder")

        self.dec: Optional[CWFormerStreamingONNX] = None
        self.mel_disp: Optional[MelComputer] = None
        self.water: Optional[StreamingPower] = None
        self.mel_buf: Optional[np.ndarray] = None
        self.water_buf: Optional[np.ndarray] = None

        self._n_water_disp: int = 76      # set on model load
        self.sample_rate: int = 16000
        self.chunk_samples: int = 8000

        self.water_history: Optional[np.ndarray] = None
        self.mel_history: Optional[np.ndarray] = None
        self.total_mel_frames = 0

        # Each entry: (mel_frame_abs, char). Always full history (causal CTC).
        self.chars: List[Tuple[int, str]] = []
        self._chars_drawn = 0  # count appended to text widget

        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

        self._build_ui(default_model)
        self._poll_queue()

    # ---- UI construction ----

    def _build_ui(self, default_model: Optional[str]) -> None:
        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.pack(side="top", fill="x")

        ttk.Label(ctrl, text="Source:").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="File")
        sources = ["File", "Stdin"] + self._list_input_devices()
        self.source_combo = ttk.Combobox(
            ctrl, textvariable=self.source_var, values=sources,
            width=60, state="readonly",
        )
        self.source_combo.grid(row=0, column=1, sticky="we", padx=4)
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_change)

        ttk.Label(ctrl, text="File:").grid(row=1, column=0, sticky="w")
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(ctrl, textvariable=self.file_var)
        self.file_entry.grid(row=1, column=1, sticky="we", padx=4)
        self.file_browse = ttk.Button(
            ctrl, text="Browse...", command=self._on_browse_file)
        self.file_browse.grid(row=1, column=2, padx=4)

        ttk.Label(ctrl, text="Model:").grid(row=2, column=0, sticky="w")
        self.model_var = tk.StringVar(value=default_model or "")
        self.model_entry = ttk.Entry(ctrl, textvariable=self.model_var)
        self.model_entry.grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(ctrl, text="Browse...",
                   command=self._on_browse_model).grid(row=2, column=2, padx=4)

        btnf = ttk.Frame(ctrl)
        btnf.grid(row=3, column=0, columnspan=3, sticky="we", pady=(6, 0))
        self.start_btn = ttk.Button(btnf, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=2)
        self.stop_btn = ttk.Button(
            btnf, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        ttk.Button(btnf, text="Clear", command=self._on_clear).pack(
            side="left", padx=2)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(btnf, textvariable=self.status_var).pack(side="right")

        ctrl.columnconfigure(1, weight=1)

        # === Plot ===
        self.fig = Figure(figsize=(11, 6), dpi=100)
        gs = self.fig.add_gridspec(
            3, 1, height_ratios=[0.6, 3, 3], hspace=0.05,
            left=0.07, right=0.98, top=0.97, bottom=0.08,
        )
        self.ax_letters = self.fig.add_subplot(gs[0, 0])
        self.ax_water = self.fig.add_subplot(gs[1, 0], sharex=self.ax_letters)
        self.ax_mel = self.fig.add_subplot(gs[2, 0], sharex=self.ax_letters)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # === Text output ===
        textf = ttk.Frame(self.root)
        textf.pack(side="bottom", fill="x")
        ttk.Label(textf, text="Decoded text:").pack(
            side="top", anchor="w", padx=4)
        innerf = ttk.Frame(textf)
        innerf.pack(side="top", fill="x", padx=4, pady=(0, 4))
        self.text = tk.Text(innerf, height=4, wrap="word",
                            font=("Consolas", 11))
        self.text.pack(side="left", fill="x", expand=True)
        sc = ttk.Scrollbar(innerf, orient="vertical", command=self.text.yview)
        sc.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=sc.set)

        self._render_plot()  # initial blank
        self._on_source_change()

    @staticmethod
    def _list_input_devices() -> List[str]:
        try:
            import sounddevice as sd
        except ImportError:
            return []
        out: List[str] = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) >= 1:
                    out.append(f"Device {i}: {d['name']}")
        except Exception:
            pass
        return out

    def _on_source_change(self, _evt=None) -> None:
        is_file = self.source_var.get() == "File"
        state = "normal" if is_file else "disabled"
        self.file_entry.configure(state=state)
        self.file_browse.configure(state=state)

    def _on_browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[
                ("Audio", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)

    def _on_browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose ONNX model",
            filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")],
            initialdir=str(_THIS_DIR),
        )
        if path:
            self.model_var.set(path)

    # ---- Run / Stop / Clear ----

    def _on_start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        model_path = self.model_var.get().strip()
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Model", "Choose an ONNX model file.")
            return

        try:
            self._init_decoder(model_path)
        except Exception as e:
            messagebox.showerror("Model load", str(e))
            return

        src = self.source_var.get()
        self.stop_event.clear()
        self._reset_state()

        if src == "File":
            file_path = self.file_var.get().strip()
            if not file_path or not Path(file_path).exists():
                messagebox.showerror("File", "Choose an audio file.")
                return
            it = file_source(file_path, self.sample_rate, self.chunk_samples,
                             self.stop_event, realtime=True)
            label = f"File: {Path(file_path).name}"
        elif src == "Stdin":
            it = stdin_source(self.chunk_samples, self.stop_event)
            label = "Stdin"
        elif src.startswith("Device "):
            try:
                dev_id = int(src.split(":")[0].split()[1])
            except Exception:
                messagebox.showerror("Device", f"Cannot parse device: {src}")
                return
            try:
                it = device_source(self.sample_rate, self.chunk_samples,
                                   dev_id, self.stop_event)
            except ImportError:
                messagebox.showerror(
                    "Device",
                    "sounddevice not installed. Run: pip install sounddevice")
                return
            except Exception as e:
                messagebox.showerror("Device", str(e))
                return
            label = src
        else:
            messagebox.showerror("Source", f"Unknown source: {src}")
            return

        self.status_var.set(f"Running ({label})")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.source_combo.configure(state="disabled")
        self.worker = threading.Thread(
            target=self._worker_loop, args=(it,), daemon=True)
        self.worker.start()

    def _on_stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def _on_clear(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self._reset_state()
        self.text.delete("1.0", "end")
        self._chars_drawn = 0
        self._render_plot()

    # ---- Decoder lifecycle ----

    def _init_decoder(self, model_path: str) -> None:
        self.dec = CWFormerStreamingONNX(model_path=model_path)
        self.sample_rate = self.dec.sample_rate
        self.chunk_samples = self.dec._chunk_samples
        cfg = self.dec.config
        self.mel_disp = MelComputer(cfg, config_dir=str(Path(model_path).parent))
        self.water = StreamingPower(n_fft=cfg["n_fft"], hop=cfg["hop_length"])

        # Crop linear power for display (drop bins above WATERFALL_MAX_HZ).
        n_freqs = cfg["n_fft"] // 2 + 1
        max_bin = int(np.ceil(WATERFALL_MAX_HZ * cfg["n_fft"] / self.sample_rate))
        self._n_water_disp = min(n_freqs, max_bin + 1)

    def _reset_state(self) -> None:
        if self.dec is not None:
            self.dec.reset()
        self.mel_buf = None
        self.water_buf = None
        self.total_mel_frames = 0
        self.chars = []
        self._chars_drawn = 0

        n_mels = self.dec.config["n_mels"] if self.dec else 40
        # Initialize histories with floor (renders as dark / no-data).
        self.water_history = np.full(
            (self._n_water_disp, WINDOW_FRAMES), -10.0, dtype=np.float32)
        self.mel_history = np.full(
            (n_mels, WINDOW_FRAMES), -10.0, dtype=np.float32)

    # ---- Worker thread ----

    def _worker_loop(self, audio_iter: Iterator[np.ndarray]) -> None:
        assert self.dec is not None and self.mel_disp is not None and self.water is not None
        try:
            for chunk in audio_iter:
                if self.stop_event.is_set():
                    break

                mel_frames, self.mel_buf = self.mel_disp.compute_streaming(
                    chunk, self.mel_buf)
                water_frames, self.water_buf = self.water.compute_streaming(
                    chunk, self.water_buf)
                # Drive the model. feed_audio does NOT normalize (file is
                # pre-normalized in file_source; live owns its own gain).
                self.dec.feed_audio(chunk)

                if self.dec._all_log_probs:
                    all_lp = np.concatenate(self.dec._all_log_probs, axis=0)
                    chars = greedy_with_frames(all_lp)
                else:
                    chars = []

                self.event_queue.put(
                    ("update", mel_frames, water_frames, chars))
        except Exception as e:
            self.event_queue.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            try:
                if self.dec is not None:
                    self.dec.flush()
                    if self.dec._all_log_probs:
                        all_lp = np.concatenate(self.dec._all_log_probs, axis=0)
                        chars = greedy_with_frames(all_lp)
                        self.event_queue.put(
                            ("update",
                             np.zeros((0, self.dec.config["n_mels"]), dtype=np.float32),
                             np.zeros((0, self._n_water_disp), dtype=np.float32),
                             chars))
            except Exception:
                pass
            self.event_queue.put(("done",))

    # ---- Main-thread event handling ----

    def _poll_queue(self) -> None:
        try:
            ev = None
            # Coalesce: only render once per poll, but apply all data.
            pending_chars: Optional[list] = None
            mel_accum: List[np.ndarray] = []
            water_accum: List[np.ndarray] = []
            while True:
                ev = self.event_queue.get_nowait()
                kind = ev[0]
                if kind == "update":
                    _, mel_frames, water_frames, chars = ev
                    if mel_frames.ndim == 3:
                        mel_frames = mel_frames[0]
                    if mel_frames.shape[0] > 0:
                        mel_accum.append(mel_frames)
                    if water_frames.shape[0] > 0:
                        water_accum.append(water_frames)
                    pending_chars = chars
                elif kind == "done":
                    self._on_worker_done()
                elif kind == "error":
                    self._on_worker_error(ev[1])
        except queue.Empty:
            pass

        if mel_accum or water_accum or pending_chars is not None:
            mel_concat = (np.concatenate(mel_accum, axis=0)
                          if mel_accum else np.zeros((0, 0), dtype=np.float32))
            water_concat = (np.concatenate(water_accum, axis=0)
                            if water_accum else np.zeros((0, 0), dtype=np.float32))
            self._ingest(mel_concat, water_concat,
                         pending_chars if pending_chars is not None else [])
            self._render_plot()

        self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _on_worker_done(self) -> None:
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.source_combo.configure(state="readonly")
        self.status_var.set("Idle")

    def _on_worker_error(self, msg: str) -> None:
        self._on_worker_done()
        messagebox.showerror("Decoder error", msg)

    def _ingest(
        self,
        mel_frames: np.ndarray,
        water_frames: np.ndarray,
        chars: List[Tuple[str, int]],
    ) -> None:
        if self.water_history is None or self.mel_history is None:
            return
        n = min(
            mel_frames.shape[0] if mel_frames.size else 0,
            water_frames.shape[0] if water_frames.size else 0,
        )
        if n > 0:
            mel_t = mel_frames[:n].T
            water_t = water_frames[:n, : self._n_water_disp].T
            self._roll(self.mel_history, mel_t)
            self._roll(self.water_history, water_t)
            self.total_mel_frames += n

        # Causal CTC: 'chars' is monotonic across calls. Append-only update.
        if len(chars) > len(self.chars):
            new = chars[len(self.chars):]
            # 2x subsampling: ctc_frame * 2 ≈ mel_frame.
            for ch, ctc in new:
                self.chars.append((ctc * 2, ch))

        # Append-only update of the text widget.
        if len(self.chars) > self._chars_drawn:
            new_text = "".join(
                ch for _, ch in self.chars[self._chars_drawn:])
            self.text.insert("end", new_text)
            self.text.see("end")
            self._chars_drawn = len(self.chars)

    @staticmethod
    def _roll(history: np.ndarray, new_cols: np.ndarray) -> None:
        n = new_cols.shape[1]
        w = history.shape[1]
        if n >= w:
            history[:, :] = new_cols[:, -w:]
        elif n > 0:
            history[:, :-n] = history[:, n:]
            history[:, -n:] = new_cols

    # ---- Plot rendering ----

    def _render_plot(self) -> None:
        # Live-scrolling window: newest data at the right edge.
        t_end = self.total_mel_frames
        t_start = t_end - WINDOW_FRAMES  # may be negative during warm-up

        cfg = self.dec.config if self.dec is not None else {
            "n_fft": 400, "f_min": 200.0, "f_max": 1400.0, "n_mels": 40,
        }
        n_fft = cfg["n_fft"]
        sr = self.sample_rate
        freq_max_disp = (self._n_water_disp - 1) * sr / n_fft

        # === Waterfall ===
        self.ax_water.cla()
        if self.water_history is not None:
            self.ax_water.imshow(
                self.water_history,
                aspect="auto", origin="lower",
                cmap="magma",
                vmin=-10.0, vmax=4.0,
                extent=[t_start, t_end, 0.0, freq_max_disp],
                interpolation="nearest",
            )
        self.ax_water.set_ylabel("Audio (Hz)")
        self.ax_water.tick_params(labelbottom=False)

        # === Mel ===
        self.ax_mel.cla()
        if self.mel_history is not None:
            self.ax_mel.imshow(
                self.mel_history,
                aspect="auto", origin="lower",
                cmap="viridis",
                vmin=-10.0, vmax=2.0,
                extent=[t_start, t_end, cfg["f_min"], cfg["f_max"]],
                interpolation="nearest",
            )
        self.ax_mel.set_ylabel(
            f"Mel ({int(cfg['f_min'])}-{int(cfg['f_max'])} Hz)")
        self.ax_mel.set_xlabel(
            "Time (mel frames, 10 ms each)  -->  newest")

        # === Letters lane ===
        self.ax_letters.cla()
        self.ax_letters.set_yticks([])
        self.ax_letters.set_ylim(0, 1)
        for sp in ("top", "right", "left"):
            self.ax_letters.spines[sp].set_visible(False)
        self.ax_letters.tick_params(labelbottom=False, bottom=False)

        # Vertical lines + letters in the visible window.
        # Iterate from the back since chars is sorted by frame; can early-out.
        i = len(self.chars) - 1
        while i >= 0:
            mel_frame_abs, ch = self.chars[i]
            if mel_frame_abs < t_start:
                break
            if mel_frame_abs <= t_end:
                # Stagger heights slightly to reduce overlap at high WPM.
                y = 0.6 if (i % 2 == 0) else 0.3
                self.ax_letters.text(
                    mel_frame_abs, y, ch,
                    ha="center", va="center",
                    fontsize=10, fontfamily="monospace", color="black",
                )
                self.ax_letters.axvline(
                    mel_frame_abs, color="black", alpha=0.5, linewidth=0.6)
                self.ax_water.axvline(
                    mel_frame_abs, color="white", alpha=0.4, linewidth=0.6)
                self.ax_mel.axvline(
                    mel_frame_abs, color="white", alpha=0.4, linewidth=0.6)
            i -= 1

        self.ax_letters.set_xlim(t_start, t_end)
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_model() -> Optional[str]:
    deploy = Path(__file__).resolve().parent
    for name in ("cwformer_streaming_int8.onnx",
                 "cwformer_streaming_fp32.onnx"):
        p = deploy / name
        if p.exists():
            return str(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("CWformer GUI: cross-platform decoder with live "
                     "waterfall + mel display."),
    )
    parser.add_argument(
        "--model", default=None,
        help=("ONNX model path (default: deploy/cwformer_streaming_int8.onnx "
              "if present, else fp32)"))
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1200x800")
    CWformerGUI(root, args.model or _default_model())
    root.mainloop()


if __name__ == "__main__":
    main()
