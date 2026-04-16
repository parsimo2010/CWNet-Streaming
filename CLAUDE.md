# CWNet-Streaming — Claude Reference Overview

## Project Intent & Goals

CWNet-Streaming is a **causal streaming** neural Morse code (CW) decoder, evolved from the bidirectional CW-Former in [CWNet](https://github.com/parsimo2010/CWNet). It uses a fully causal Conformer architecture (~19.5M params) with CTC loss that processes audio left-to-right with no bidirectional attention.

**Why this exists:** The original CW-Former achieved < 5% CER during training but degraded in real-world streaming due to window stitching artifacts. The sliding-window approach forced a trade-off between accuracy (large windows) and latency (small windows). This project eliminates stitching entirely by using a causal model with KV cache state carry-forward.

**Design philosophy:** Process audio causally (each frame sees only past context), carry state between processing chunks via KV cache and conv buffers, and use a commitment delay to avoid premature character emission. Never emit a character until the inter-character space (ICS) is confirmed — many characters share element prefixes (E=`.`, I=`..`, S=`...`, H=`....`, 5=`.....`).

**Target performance:** 15-40 WPM primary window, any key type (straight, bug, paddle, cootie), SNR > 5-8 dB. < 2.5s latency from audio to character emission. Desktop CPU/GPU deployment.

**Architecture model:** Causal Conformer, following NVIDIA FastConformer (streaming mode) and Google streaming Conformer. CTC loss with greedy decoding.

**See `PLAN.md`** for the full implementation plan, detailed file-by-file change descriptions, training strategy, and verification plan.

---

## Architecture

```
Audio (16 kHz mono, streaming chunks)
  -> MelFrontend: incremental log-mel spectrogram (40 bins, 200-1400 Hz, 25ms/10ms)
     + SpecAugment (training only)
  -> Causal ConvSubsampling: 2x time reduction (left-pad only in time)
     -> 50 fps (20ms per CTC frame)
  -> Causal ConformerEncoder: 12 blocks (d=256, 4 heads, conv kernel=31)
     - Self-attention: fully causal (is_causal=True during training, KV cache during inference)
     - Convolution: causal depthwise (left-pad=30, right-pad=0, 620ms receptive field)
     - Feed-forward: Macaron-style half-step (pointwise, unchanged from CWNet)
  -> Linear CTC head -> log_softmax -> incremental greedy decode with commitment delay -> text
```

**Key difference from CWNet:** Attention is fully causal (each frame only attends to past frames), convolutions are left-padded only, and inference uses KV cache + conv buffers for state continuity between chunks. No window stitching.

**Commitment delay:** Characters are not emitted until the ICS after them is confirmed. Configurable up to ~2 seconds. At 25 WPM, most characters commit within 0.5-1s. This prevents premature emission on ambiguous element prefixes.

---

## File Map & Key Functions

### Infrastructure (unchanged from CWNet)
- `config.py` — `MorseConfig`, `TrainingConfig`, `create_default_config(scenario)`. **sample_rate = 16000**.
- `vocab.py` — CTC vocabulary (52 classes). `encode(text)`, `decode(indices)`, `decode_ctc(log_probs)`.
- `morse_table.py` — ITU Morse code table + binary trie.
- `morse_generator.py` — Synthetic training data. `generate_sample(config)` -> `(audio_f32, text, metadata)`. All augmentations: AGC, QSB, QRM, QRN, bandpass, HF noise, key types, timing jitter, speed drift.
- `qso_corpus.py` — `QSOCorpusGenerator` for realistic ham radio QSO text.
- `deploy/ctc_decode.py` — Pure-numpy CTC beam search with trigram LM.

### neural_decoder/ — Causal CW-Former

#### Model
- `cwformer.py` — `CWFormer` (~19.5M params): MelFrontend -> Causal ConvSubsampling -> Causal ConformerEncoder -> CTC head.
  - `forward()` — Training: full sequence with causal attention. Input: `(audio, audio_lengths)` -> `(log_probs, output_lengths)`.
  - `forward_streaming(mel_chunk, state)` — Inference: single chunk with KV cache + conv buffers. Returns `(log_probs, new_state)`.
  - `ConvSubsampling` — Causal 2x time reduction (left-pad=2, right-pad=0 in time for both Conv2d layers).
- `conformer.py` — Causal Conformer blocks.
  - `ConformerMHA` — Causal self-attention. Training: `is_causal=True` in SDPA. Inference: KV cache concatenation + causal mask within chunk. RoPE with position offset.
  - `ConvolutionModule` — Causal depthwise conv (left-pad=30, right-pad=0). Inference: conv buffer carry-forward.
  - `ConformerEncoder` — Threads KV caches and conv buffers through 12 blocks.
- `rope.py` — Rotary Position Embeddings with `offset` parameter for KV cache positions.
- `mel_frontend.py` — `MelFrontend` with `compute_streaming(audio_chunk, stft_buffer)` for incremental mel computation. STFT overlap buffer of 240 samples.

#### Training
- `dataset_audio.py` — `AudioDataset`: streaming IterableDataset (unchanged from CWNet).
- `train_cwformer.py` — Training loop. Micro-batch 8, effective batch 64 via gradient accumulation. Causal attention active during training. Supports optional streaming validation.

#### Inference
- `inference_cwformer.py` — `CWFormerStreamingDecoder`: chunk-based streaming with state carry-forward. No windows, no stitching. ICS-confirmed commitment delay (~2s). Methods: `feed_audio()`, `get_full_text()`, `flush()`, `decode_file()`, `decode_audio()`.

### Deployment
- `quantize_cwformer.py` — Streaming ONNX export with state I/O (KV caches + conv buffers as explicit input/output tensors). INT8 dynamic quantization.
- `deploy/inference_onnx.py` — `CWFormerStreamingONNX`: standalone ONNX runtime inference with streaming state management. Supports file, device, and stdin input.

### Benchmarking
- `benchmark_cwformer.py` — Structured benchmark across SNR, WPM, key types. `--streaming` flag.
- `benchmark_random_sweep.py` — Random parameter sweep benchmark. `--streaming` flag.

---

## Causal Streaming vs CWNet Bidirectional

| Aspect | CWNet (bidirectional) | CWNet-Streaming (causal) |
|--------|----------------------|--------------------------|
| Attention | Full bidirectional within window | Fully causal (past only) |
| Convolution | Symmetric padding (pad=15 each side) | Left-only padding (pad=30, 0) |
| Inference | Fixed windows + stitching | Chunk-based with KV cache, no stitching |
| Latency | 3-16s (stride + window) | < 2.5s (chunk + commitment delay) |
| State | Stateless per window | Stateful: KV cache + conv buffers |
| ONNX I/O | mel -> log_probs | mel + state_in -> log_probs + state_out |

---

## Curriculum Learning

Same as CWNet — unchanged.

| Stage | SNR | WPM | AGC | QSB | Key Types | Audio Augmentations |
|-------|-----|-----|-----|-----|-----------|---------------------|
| clean | 15-40 dB | 10-40 | 30% | 0% | 20/20/60/0 S/B/P/C | 10% Farnsworth, 50% bandpass |
| moderate | 8-35 dB | 8-45 | 50% | 25% | 25/25/35/15 S/B/P/C | 20% Farnsworth, 15% QRM, 70% bandpass |
| full | 3-30 dB | 5-50 | 70% | 50% | 30/30/20/20 S/B/P/C | 25% Farnsworth, 30% QRM, 90% bandpass |

---

## Performance Targets
- Primary window (15-40 WPM, any key type, SNR > 8 dB): < 5% CER goal
- Extended (10-45 WPM, moderate timing variance): < 8% CER goal
- Latency: < 2.5s from audio input to character emission
- Real-time factor: < 0.1 (10x faster than real-time on desktop CPU)

---

## Things to Keep in Mind

1. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally.
2. **2x subsampling gives 50 fps (20ms per CTC frame)** — resolves dits up to 40+ WPM.
3. **Causal attention** — `is_causal=True` during training, KV cache + causal mask during inference. Never let a frame see future audio.
4. **Causal convolution** — left-pad only (pad=30 for kernel=31). During inference, maintain a 30-frame conv buffer per layer.
5. **KV cache** — grows per chunk during inference. Trim to 1500 frames (30s) to cap memory. Position offset tracks absolute position for RoPE.
6. **Commitment delay** — don't emit characters until ICS is confirmed (~2s max wait). This avoids premature decoding of ambiguous element prefixes (E/I/S/H/5 all start with dit).
7. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]`.
8. **Persistent worker RNG** — use `np.random.default_rng()` (OS entropy), not `worker_info.seed`.
9. **DataLoader tuning** — `persistent_workers=True`, `prefetch_factor=4`. Audio generation is the CPU bottleneck.
10. **Training uses full sequences with causal mask** — no chunking during training. Inference chunks are for efficiency/latency, not for training.
11. **Weights from CWNet load directly** — tensor shapes are identical. Only runtime behavior (masking, padding) differs.
12. **ONNX state I/O** — 36 state tensors per layer (KV K, KV V, conv buffer) + 2 subsample buffers + position offset. Use per-layer naming: `kv_k_layer0`, `kv_v_layer0`, `conv_buf_layer0`, etc.

---

## Implementation Phases

Work through these in order. Each phase should be testable independently:

1. **Core model** — `rope.py` (offset), `conformer.py` (causal attn + conv + state), `cwformer.py` (causal subsampling + `forward_streaming`). Write unit tests for streaming equivalence.
2. **Training** — `train_cwformer.py` updates. Verify training runs and loss decreases.
3. **Inference** — `inference_cwformer.py` (`CWFormerStreamingDecoder`). Integration test with synthetic audio.
4. **ONNX/Deploy** — `quantize_cwformer.py` + `deploy/inference_onnx.py`. Verify ONNX parity.
5. **Benchmarking** — `benchmark_cwformer.py` + `benchmark_random_sweep.py`. Compare CER vs CWNet bidirectional.
