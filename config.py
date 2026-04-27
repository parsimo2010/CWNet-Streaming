"""
config.py — Configuration for CWformer Morse code decoder.

Dataclasses covering audio generation and training:
  MorseConfig    — synthetic audio generation parameters
  TrainingConfig — training hyperparameters and curriculum settings

Use create_default_config(scenario) to get pre-built configs for:
  "test"     — tiny run (~5 epochs) to verify the pipeline end-to-end
  "clean"    — curriculum stage 1: high SNR, standard timing (200 epochs)
  "moderate" — curriculum stage 2: mid SNR, moderate bad-fist (300 epochs)
  "full"     — curriculum stage 3: low SNR, extreme bad-fist (500 epochs)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# MorseConfig — audio generation
# ---------------------------------------------------------------------------

@dataclass
class MorseConfig:
    """Synthetic Morse audio generation parameters.

    Timing ratios (dah_dit, ics, iws) are sampled independently per sample
    to cover the full range of real-world operator styles.
    """

    # Internal sample rate; all audio inputs are resampled to this at inference
    sample_rate: int = 16000

    # WPM range
    min_wpm: float = 10.0
    max_wpm: float = 40.0

    # Tone carrier frequency range (Hz)
    tone_freq_min: float = 500.0
    tone_freq_max: float = 900.0

    # Slow sinusoidal frequency drift (Hz peak deviation) — simulates VFO drift
    tone_drift: float = 3.0

    # SNR (dB) — measured against full-band white AWGN
    min_snr_db: float = 15.0
    max_snr_db: float = 40.0

    # Timing jitter: fraction of unit duration (std dev of Gaussian perturbation)
    # Actual per-sample jitter is drawn uniformly in [timing_jitter, timing_jitter_max]
    timing_jitter: float = 0.0
    timing_jitter_max: float = 0.05

    # Dah/dit ratio (ITU standard = 3.0; bad-fist operators can go down to 1.5)
    dah_dit_ratio_min: float = 2.5
    dah_dit_ratio_max: float = 3.5

    # Inter-character space factor (× standard 3-dit gap)
    # 1.0 = standard; <1.0 = compressed; >1.0 = expanded
    ics_factor_min: float = 0.8
    ics_factor_max: float = 1.2

    # Inter-word space factor (× standard 7-dit gap)
    iws_factor_min: float = 0.8
    iws_factor_max: float = 1.5

    # Text length range (characters, including spaces)
    min_chars: int = 20
    max_chars: int = 120

    # Signal amplitude variation across samples
    signal_amplitude_min: float = 0.5
    signal_amplitude_max: float = 0.9

    # AGC simulation — noise-floor modulation matching real HF radio AGC.
    # During marks the AGC reduces gain → background noise is suppressed.
    # During spaces the AGC releases → noise rises to full level over release_ms.
    # This creates the characteristic noise-floor drift seen between elements in
    # real recordings.  Noise is modulated *before* the IF filter so the effect
    # appears in the feature extractor's noise estimate.
    agc_probability: float = 0.0        # fraction of samples with AGC enabled
    agc_attack_ms: float = 50.0         # gain reduction time constant (ms)
    agc_release_ms: float = 400.0       # gain recovery time constant (ms)
    agc_depth_db_min: float = 6.0       # noise suppression at peak mark (dB, min)
    agc_depth_db_max: float = 15.0      # noise suppression at peak mark (dB, max)

    # QSB — slow sinusoidal signal fading within a sample (0.05–0.3 Hz).
    # Captures mark-to-mark amplitude variation from propagation.
    qsb_probability: float = 0.0
    qsb_depth_db_min: float = 3.0      # peak-to-peak fading range (dB, min)
    qsb_depth_db_max: float = 10.0     # peak-to-peak fading range (dB, max)

    # Key type weights: (straight_key, bug, paddle, cootie) probabilities.
    # Straight key: per-character speed variation, high jitter on all elements.
    # Bug (semi-automatic): consistent dits, variable dahs + spacing.
    # Paddle (electronic keyer): consistent elements, variable spacing only.
    # Cootie (sideswiper): alternating contacts, symmetric but high jitter,
    #   no inherent dit/dah length distinction — operator must time everything.
    key_type_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    # Speed drift: slow WPM variation within a single transmission.
    # Fraction of base unit_dur (e.g. 0.15 = ±15%).  0.0 = constant speed.
    speed_drift_max: float = 0.0

    # Farnsworth timing: characters sent at a faster speed (char_wpm) but
    # inter-character and inter-word spaces stretched to achieve a slower
    # effective speed (the configured WPM).  Common in CW training and some
    # operators' natural style.  0.0 = disabled, otherwise probability of
    # applying Farnsworth timing to a given sample.
    farnsworth_probability: float = 0.0
    # The character speed multiplier: char_wpm = wpm * farnsworth_char_speed_mult.
    # E.g. 1.5 means characters sent 50% faster than overall effective speed.
    farnsworth_char_speed_min: float = 1.3
    farnsworth_char_speed_max: float = 2.0

    # Keying waveform shaping: rise/fall time range for mark envelopes.
    # Real transmitters have 2-8 ms rise/fall; 0 ms = hard keying (unrealistic).
    # The default 5 ms is a good middle ground.
    rise_time_ms_min: float = 3.0
    rise_time_ms_max: float = 8.0

    # QRM — interfering CW signals at nearby frequencies.
    # Simulates other operators transmitting on adjacent frequencies.
    qrm_probability: float = 0.0         # fraction of samples with QRM
    qrm_count_min: int = 1               # min number of interferers
    qrm_count_max: int = 3               # max number of interferers
    qrm_freq_offset_min: float = 100.0   # min frequency offset from target (Hz)
    qrm_freq_offset_max: float = 500.0   # max frequency offset from target (Hz)
    qrm_amplitude_min: float = 0.1       # min amplitude relative to main signal
    qrm_amplitude_max: float = 0.8       # max amplitude relative to main signal

    # QRN — impulsive atmospheric noise (static crashes from lightning).
    # Poisson-distributed impulses with random duration and amplitude.
    qrn_probability: float = 0.0         # fraction of samples with QRN
    qrn_rate_min: float = 0.5            # min impulse rate (per second)
    qrn_rate_max: float = 5.0            # max impulse rate (per second)
    qrn_duration_ms_min: float = 1.0     # min impulse duration (ms)
    qrn_duration_ms_max: float = 50.0    # max impulse duration (ms)
    qrn_amplitude_min: float = 0.3       # min impulse amplitude (relative to signal)
    qrn_amplitude_max: float = 2.0       # max impulse amplitude (relative to signal)

    # Receiver bandpass filter — simulates a real CW filter (200-500 Hz BW).
    # Applied after all signal mixing, before normalisation.
    # Real radios always have a filter; probability should be high (0.5-1.0).
    bandpass_probability: float = 0.0     # fraction of samples with bandpass
    bandpass_bw_min: float = 200.0        # min filter bandwidth (Hz)
    bandpass_bw_max: float = 500.0        # max filter bandwidth (Hz)
    bandpass_order_min: int = 4           # min Butterworth filter order
    bandpass_order_max: int = 4           # max Butterworth filter order (sampled per sample)

    # Real HF noise — mix recorded HF band noise instead of/with AWGN.
    # Bridges the synthetic-to-real gap by using actual band characteristics.
    hf_noise_probability: float = 0.0     # fraction of samples using real HF noise
    hf_noise_dir: str = "recordings"      # directory containing noise WAV files
    hf_noise_mix_ratio: float = 0.7       # fraction of noise that is real HF (rest is AWGN)

    # Multi-operator speed change — abrupt WPM changes between words.
    # Simulates operator changes on multi-op stations or natural speed variation.
    multi_op_probability: float = 0.0     # fraction of samples with speed changes
    multi_op_speed_change_min: float = 0.7   # min speed multiplier at change point
    multi_op_speed_change_max: float = 1.4   # max speed multiplier at change point

    # ---- Multi-segment (multiple SEQUENTIAL senders within one sample) ----
    # Each sample is composed of N independent operator transmissions
    # separated by silent gaps. One radio (one bandpass, one AGC, one
    # noise floor) listening to multiple ops. Trains the streaming KV
    # cache to release context after a gap rather than locking onto one
    # signal -- this is the fix for the state-drift failure mode seen
    # on long real-world recordings.
    multi_segment_probability: float = 0.0
    multi_segment_count_min: int = 1
    multi_segment_count_max: int = 4

    # Edge-silence distribution (seconds). Spans 0-10 s with an
    # exponential-tailed bias toward short silences so the model sees
    # a wide range of "leading dead air" without losing the typical
    # short-edge case.
    multi_segment_leading_silence_max_sec: float = 10.0
    multi_segment_trailing_silence_max_sec: float = 10.0
    multi_segment_edge_silence_scale_sec: float = 2.0

    # Gap distribution (seconds). The short-gap mass forces fast cache
    # release; the long tail (up to ~15 s) covers extended operator
    # pauses so the model treats long silence as a normal training
    # signal rather than a positional cue.
    multi_segment_gap_min: float = 0.3
    multi_segment_gap_max: float = 5.0
    multi_segment_gap_short_prob: float = 0.6   # prob gap <= 1.5 s
    multi_segment_gap_long_max_sec: float = 15.0

    # Per-segment short-burst character budget. All segments except the
    # last are sized as a small log-uniform burst in this range; the
    # last segment fills whatever budget remains. Combined with the
    # post-build random shuffle, this removes the predictable boundary
    # times that fall out of equal slot division.
    multi_segment_short_burst_chars_min: int = 1
    multi_segment_short_burst_chars_max: int = 15

    # Pitch contrast tiers (Hz) between adjacent segments. Probabilities
    # for the four tiers; residual goes to the widest tier. Same-bin
    # (0-10 Hz) is the hardest case -- only the fist distinguishes ops.
    multi_segment_pitch_same_bin_prob: float = 0.35   # 0-10 Hz
    multi_segment_pitch_near_prob: float = 0.30       # 10-50 Hz
    multi_segment_pitch_medium_prob: float = 0.25     # 50-200 Hz
    # remainder -> 200-500 Hz

    # WPM contrast tiers between adjacent segments.
    multi_segment_wpm_match_prob: float = 0.30   # within 1 WPM (zero-beat)
    multi_segment_wpm_close_prob: float = 0.35   # within 5 WPM
    multi_segment_wpm_diff_prob: float = 0.25    # within 15 WPM
    # remainder -> wider

    multi_segment_same_key_type_prob: float = 0.50

    # Noise consistency: most samples represent one radio with stable
    # band conditions; only a minority simulate user retuning mid-recording.
    multi_segment_noise_change_prob: float = 0.15

    # Per-segment relative amplitude jitter (dB below base, one-sided).
    # Models fading / different operator power. Effective per-segment
    # SNR varies because amplitude varies, while the noise floor stays
    # uniform across the sample.
    multi_segment_amplitude_jitter_db: float = 6.0

    # ---- Tier 3: short-burst sender alternation (a.k.a. "fist burst") ----
    # Each segment is a short multi-character burst from one operator,
    # rendered with independently sampled (pitch, WPM, key, fist) drawn
    # from a NARROW distribution centred on a session nominal value.
    # Narrow pitch jitter forces the model to discriminate operators by
    # FIST inside the same mel bin. Real-world fist changes inside a
    # transmission are rarely letter-by-letter; they're typically
    # multi-character runs (handoffs, brief jump-ins). Burst sizes
    # default to 1-1 for backward compatibility (= legacy letter
    # alternation). Bump to 2-5 in the full curriculum to match the
    # realistic distribution.
    letter_alternation_probability: float = 0.0
    letter_alternation_pitch_jitter_hz: float = 15.0
    letter_alternation_gap_min: float = 0.18
    letter_alternation_gap_max: float = 0.50
    letter_alternation_count_min: int = 8
    letter_alternation_count_max: int = 30
    # Characters per burst. 1/1 = legacy per-letter alternation;
    # 2/5 = realistic multi-character bursts.
    letter_alternation_chars_per_burst_min: int = 1
    letter_alternation_chars_per_burst_max: int = 1

    # Random input gain (dB), applied AFTER peak-normalisation in
    # generate_sample(). Drawn log-uniformly in [lo, hi] dB per sample and
    # multiplied onto the waveform (then clipped to [-1, 1]). Teaches the
    # model to handle inputs that aren't peak-normalised, which matches
    # the real streaming inference regime where per-chunk peak varies.
    # (0.0, 0.0) disables the augmentation.
    input_gain_db_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MorseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# TrainingConfig — training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 300

    # Synthetic samples generated per epoch (train / validation)
    samples_per_epoch: int = 5000
    val_samples: int = 500

    # DataLoader worker processes (0 = main process only)
    num_workers: int = 4

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50           # batches between mid-epoch loss prints

    # Streaming-mode validation: run val audio through CWFormerStreamingDecoder
    # every N epochs and log val_cer_stream alongside val_cer_full. Drift of
    # > ~1% absolute CER on converged in-distribution audio is a regression
    # signal (cuDNN kernel heuristics + chunked state plumbing). 0 disables
    # the feature; validation stays full-forward only.
    stream_val_every_n_epochs: int = 0
    stream_val_chunk_ms: int = 500
    stream_val_max_cache_sec: float = 30.0
    # Cap on how many val samples go through the streaming path per eval.
    # Per-sample streaming is serial (each chunk waits on the prior state),
    # so doing all 5000 val samples turns a 30 s full-forward eval into a
    # multi-hour wait. 50 samples × ~50 chars each ≈ 2500-char CER base —
    # enough resolution to spot >~1% drift between paths.
    stream_val_samples: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Full pipeline configuration (generation + training)."""

    morse: MorseConfig = field(default_factory=MorseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return {
            "morse":    self.morse.to_dict(),
            "training": self.training.to_dict(),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        # Older saved configs may still include "feature" and "model"
        # sections from the pre-cleanup schema. They are ignored here.
        return cls(
            morse=MorseConfig.from_dict(d.get("morse", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
        )


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def create_default_config(scenario: str = "clean") -> Config:
    """Return a pre-configured Config for the given training scenario.

    Scenarios
    ---------
    test     — 5 epochs, tiny epoch size; verifies the full pipeline
    clean    — 200 epochs; high SNR, near-standard timing (curriculum stage 1)
    moderate — 300 epochs; mid SNR, moderate bad-fist (curriculum stage 2)
    full     — 500 epochs; low SNR, extreme bad-fist (curriculum stage 3)
    """
    cfg = Config()

    if scenario == "test":
        cfg.morse.min_snr_db = 20.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 15.0
        cfg.morse.max_wpm = 25.0
        cfg.morse.min_chars = 15
        cfg.morse.max_chars = 40
        cfg.morse.dah_dit_ratio_min = 2.8
        cfg.morse.dah_dit_ratio_max = 3.2
        cfg.morse.ics_factor_min = 0.9
        cfg.morse.ics_factor_max = 1.1
        cfg.morse.iws_factor_min = 0.9
        cfg.morse.iws_factor_max = 1.1
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.02
        cfg.morse.tone_drift = 1.0
        cfg.training.num_epochs = 5
        cfg.training.samples_per_epoch = 500
        cfg.training.val_samples = 100
        cfg.training.num_workers = 0
        cfg.training.batch_size = 16
        cfg.training.learning_rate = 5e-4

    elif scenario == "clean":
        cfg.morse.min_snr_db = 15.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 30
        cfg.morse.max_chars = 150
        cfg.morse.dah_dit_ratio_min = 2.5
        cfg.morse.dah_dit_ratio_max = 3.5
        cfg.morse.ics_factor_min = 0.8
        cfg.morse.ics_factor_max = 1.2
        cfg.morse.iws_factor_min = 0.8
        cfg.morse.iws_factor_max = 1.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.05
        cfg.morse.tone_drift = 3.0
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 200
        cfg.training.samples_per_epoch = 100000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (mild -- model learns basic task first)
        cfg.morse.agc_probability = 0.2
        # qsb_probability stays off for the clean stage
        cfg.morse.qsb_probability = 0.0
        # Key type: mostly paddles (easiest) for clean stage; no cootie yet
        cfg.morse.key_type_weights = (0.20, 0.20, 0.60, 0.0)
        # Farnsworth: mild introduction (10% of samples, mild stretch)
        cfg.morse.farnsworth_probability = 0.10
        cfg.morse.farnsworth_char_speed_min = 1.2
        cfg.morse.farnsworth_char_speed_max = 1.5
        # Bandpass filter: half of samples, wide filter, gentle slopes
        cfg.morse.bandpass_probability = 0.50
        cfg.morse.bandpass_bw_min = 400.0
        cfg.morse.bandpass_bw_max = 500.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 6
        # Real HF noise: mild introduction (15% of samples)
        cfg.morse.hf_noise_probability = 0.15
        cfg.morse.hf_noise_mix_ratio = 0.5
        # Input gain: disabled for the clean stage (waveform stays peak-normalised)
        cfg.morse.input_gain_db_range = (0.0, 0.0)

    elif scenario == "moderate":
        cfg.morse.min_snr_db = 5.0
        cfg.morse.max_snr_db = 35.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 25
        cfg.morse.max_chars = 175
        cfg.morse.dah_dit_ratio_min = 1.8
        cfg.morse.dah_dit_ratio_max = 3.8
        cfg.morse.ics_factor_min = 0.6
        cfg.morse.ics_factor_max = 1.6
        cfg.morse.iws_factor_min = 0.6
        cfg.morse.iws_factor_max = 2.0
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.15
        cfg.morse.tone_drift = 4.0
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 300
        # samples_per_epoch reduced 75k -> 45k -> 36k. The 75 -> 45
        # accounted for ~1.6x longer avg sample duration when multi-
        # segment is active (40% multi-seg @ avg ~45 s vs single-seg
        # avg ~17 s). The 45 -> 36 accounts for the SWA training cost:
        # the explicit attn_mask blocks the Flash kernel and falls back
        # to memory-efficient SDPA, which costs ~20% per attention call.
        # The new letter-alt-at-moderate addition is small (5% prob at
        # ~22 s avg) and shifts the mean by <1%; no compensation needed.
        cfg.training.samples_per_epoch = 36000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (moderate strength)
        cfg.morse.agc_probability = 0.4
        cfg.morse.agc_depth_db_max = 18.0
        cfg.morse.qsb_probability = 0.25
        cfg.morse.qsb_depth_db_max = 12.0
        # Key type: balanced mix, introduce cootie
        cfg.morse.key_type_weights = (0.25, 0.25, 0.35, 0.15)
        # Speed drift: mild ±8% WPM variation
        cfg.morse.speed_drift_max = 0.08
        # Farnsworth: moderate (20% of samples)
        cfg.morse.farnsworth_probability = 0.20
        cfg.morse.farnsworth_char_speed_min = 1.3
        cfg.morse.farnsworth_char_speed_max = 1.8
        # QRM: light introduction (15% of samples, 1-2 interferers)
        cfg.morse.qrm_probability = 0.15
        cfg.morse.qrm_count_max = 2
        cfg.morse.qrm_amplitude_max = 0.5
        # QRN: light introduction (15% of samples)
        cfg.morse.qrn_probability = 0.15
        cfg.morse.qrn_rate_max = 3.0
        cfg.morse.qrn_amplitude_max = 1.0
        # Bandpass filter: narrower filters, sharper slopes
        cfg.morse.bandpass_probability = 0.60
        cfg.morse.bandpass_bw_min = 250.0
        cfg.morse.bandpass_bw_max = 500.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 8
        # Real HF noise: moderate (30% of samples)
        cfg.morse.hf_noise_probability = 0.30
        cfg.morse.hf_noise_mix_ratio = 0.6
        # Multi-operator: disabled at moderate stage
        cfg.morse.multi_op_probability = 0.0
        cfg.morse.multi_op_speed_change_min = 0.8
        cfg.morse.multi_op_speed_change_max = 1.3
        # Input gain: ±6 dB log-uniform per sample
        cfg.morse.input_gain_db_range = (-6.0, 6.0)
        # Multi-segment (multiple sequential senders per sample). Trains
        # the streaming KV cache to release context after a gap rather
        # than locking onto one signal -- the fix for state drift on
        # long real-world sessions. Edge silence spans 0-10 s; gaps
        # span up to 15 s. Segment durations are randomized (short
        # bursts + remainder, post-build shuffle) so boundary times do
        # not cluster at predictable positions. Run with
        # --max-audio-sec 30.
        cfg.morse.multi_segment_probability = 0.40
        cfg.morse.multi_segment_count_min = 1
        cfg.morse.multi_segment_count_max = 3
        # Short-burst alternation: enabled at moderate with FIXED 3-char
        # bursts (3/3) so the model first learns the boundary pattern
        # without burst-length variability. Full stage widens to 2-5 for
        # variety. Same gap range as full so the inter-burst silence
        # distribution stays consistent across stages.
        cfg.morse.letter_alternation_probability = 0.05
        cfg.morse.letter_alternation_chars_per_burst_min = 3
        cfg.morse.letter_alternation_chars_per_burst_max = 3
        cfg.morse.letter_alternation_gap_min = 0.30
        cfg.morse.letter_alternation_gap_max = 1.50
        cfg.morse.letter_alternation_count_min = 4
        cfg.morse.letter_alternation_count_max = 12

    elif scenario == "full":
        cfg.morse.min_snr_db = -5.0
        cfg.morse.max_snr_db = 30.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 20
        cfg.morse.max_chars = 200
        cfg.morse.dah_dit_ratio_min = 1.3
        cfg.morse.dah_dit_ratio_max = 4.0
        cfg.morse.ics_factor_min = 0.5
        cfg.morse.ics_factor_max = 2.0
        cfg.morse.iws_factor_min = 0.5
        cfg.morse.iws_factor_max = 2.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.25
        cfg.morse.tone_drift = 5.0
        # sqrt(512/128) * 6e-4 ~ 1.2e-3; rounded to 1e-3.
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 500
        # samples_per_epoch reduced 50k -> 20k -> 16k. The 50 -> 20
        # accounted for ~2.4x longer avg sample duration at full stage
        # (75% multi-seg @ avg ~50 s vs single-seg avg ~17 s). The 20
        # -> 16 accounts for the SWA training cost: explicit attn_mask
        # blocks the Flash kernel and adds ~20% per attention call,
        # which compounds at the heaviest stage where multi-segment
        # samples are longest. The letter-alt change (1-char -> 2-5
        # char bursts at the same 5% probability) shifts the average
        # sample duration by <1%; no separate compensation needed.
        cfg.training.samples_per_epoch = 16000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (full strength for curriculum stage 3)
        cfg.morse.agc_probability = 0.5
        cfg.morse.agc_depth_db_max = 22.0
        cfg.morse.qsb_probability = 0.50
        cfg.morse.qsb_depth_db_max = 24.0
        # Key type: weighted toward harder key types (straight key, bug, cootie)
        cfg.morse.key_type_weights = (0.30, 0.30, 0.20, 0.20)
        # Speed drift: ±15% WPM variation within a transmission
        cfg.morse.speed_drift_max = 0.15
        # Farnsworth: full range (25% of samples)
        cfg.morse.farnsworth_probability = 0.25
        cfg.morse.farnsworth_char_speed_min = 1.3
        cfg.morse.farnsworth_char_speed_max = 2.0
        # QRM: full strength (30% of samples, 1-3 interferers)
        cfg.morse.qrm_probability = 0.30
        cfg.morse.qrm_count_max = 3
        cfg.morse.qrm_amplitude_max = 0.8
        # QRN: full strength (25% of samples)
        cfg.morse.qrn_probability = 0.25
        cfg.morse.qrn_rate_max = 5.0
        cfg.morse.qrn_amplitude_max = 2.0
        # Bandpass filter: nearly-always present, wide BW range covering
        # everything from very narrow CW filters (100 Hz) to wide
        # SSB-style filters (700 Hz). Real receivers always have a
        # filter, so probability stays high.
        cfg.morse.bandpass_probability = 0.80
        cfg.morse.bandpass_bw_min = 100.0
        cfg.morse.bandpass_bw_max = 700.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 8
        # Real HF noise: full (50% of samples)
        cfg.morse.hf_noise_probability = 0.50
        cfg.morse.hf_noise_mix_ratio = 0.7
        # Multi-operator: full (15% of samples, wider speed range)
        cfg.morse.multi_op_probability = 0.15
        cfg.morse.multi_op_speed_change_min = 0.7
        cfg.morse.multi_op_speed_change_max = 1.4
        # Input gain: ±12 dB log-uniform per sample
        cfg.morse.input_gain_db_range = (-12.0, 12.0)
        # Multi-segment: dominant at full stage (most samples have
        # multiple senders so the model rarely sees a single coherent
        # signal context for the whole sample length). n_segments=1
        # is allowed inside the multi-segment branch so that the wide
        # 0-10 s edge-silence distribution and shuffled-burst layout
        # also apply to single-sender samples. Edge silence spans
        # 0-10 s; gaps span up to 15 s. Run with --max-audio-sec 30.
        cfg.morse.multi_segment_probability = 0.75
        cfg.morse.multi_segment_count_min = 1
        cfg.morse.multi_segment_count_max = 4
        # Tier 3: short-burst sender alternation. ~5% of multi-segment
        # samples (i.e. ~3% overall in this stage) become burst-alternated:
        # 2-5 chars from one fist, then a short-ish silence (0.3-1.5 s),
        # then 2-5 chars from another fist, etc. Narrow-pitch around a
        # session centre forces fist-only discrimination inside the same
        # mel bin. Replaces the prior per-letter alternation (which was
        # not realistic for ham radio operating practice).
        cfg.morse.letter_alternation_probability = 0.05
        cfg.morse.letter_alternation_chars_per_burst_min = 2
        cfg.morse.letter_alternation_chars_per_burst_max = 5
        cfg.morse.letter_alternation_gap_min = 0.30
        cfg.morse.letter_alternation_gap_max = 1.50
        # With 2-5 chars per burst, fewer bursts fit in the budget.
        cfg.morse.letter_alternation_count_min = 4
        cfg.morse.letter_alternation_count_max = 12

    else:
        raise ValueError(
            f"Unknown scenario: {scenario!r}.  Choose from: test, clean, moderate, full."
        )

    return cfg


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for s in ("test", "clean", "moderate", "full"):
        cfg = create_default_config(s)
        print(
            f"[{s:8s}]  SNR={cfg.morse.min_snr_db:.0f}–{cfg.morse.max_snr_db:.0f} dB  "
            f"WPM={cfg.morse.min_wpm:.0f}–{cfg.morse.max_wpm:.0f}  "
            f"dah/dit={cfg.morse.dah_dit_ratio_min:.1f}–{cfg.morse.dah_dit_ratio_max:.1f}  "
            f"epochs={cfg.training.num_epochs}"
        )
