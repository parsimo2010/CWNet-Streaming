#!/usr/bin/env python3
"""
quantize_cwformer.py — Streaming ONNX export for the causal CW-Former.

The mel frontend uses torch.stft which cannot be exported to ONNX, so the
model is split at the mel spectrogram boundary:

  - Mel spectrogram computation stays in Python/numpy (no learnable params)
  - The neural network (ConvSubsampling → Conformer → CTC head) is exported
    with explicit state I/O (KV caches + conv buffers + subsample buffers)

Usage::

    python quantize_cwformer.py --checkpoint checkpoints_cwformer/best_model.pt

Outputs:
    cwformer_streaming_fp32.onnx  — FP32 ONNX model
    cwformer_streaming_int8.onnx  — INT8 dynamically-quantized ONNX model
    mel_config.json               — Mel spectrogram parameters

ONNX model interface (streaming):
    Inputs:
        mel_chunk    (1, T_chunk, n_mels) — new mel frames
        pos_offset   (1,)                 — current position in stream
        kv_k_layer{i} (1, H, T_cached, d_k) — per-layer K cache (12 layers)
        kv_v_layer{i} (1, H, T_cached, d_k) — per-layer V cache
        conv_buf_layer{i} (1, D, kernel-1)   — per-layer conv buffer
        sub_buf1     (1, 1, 2, n_mels)       — conv1 subsample buffer
        sub_buf2     (1, C, 2, freq1)        — conv2 subsample buffer
    Outputs:
        log_probs    (T_out, 1, C)        — CTC log-probs for this chunk
        pos_offset_out (1,)               — updated position
        (all state tensors updated)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_decoder.inference_cwformer import _load_cwformer_checkpoint


# ---------------------------------------------------------------------------
# ONNX-exportable streaming core
# ---------------------------------------------------------------------------

class _CWFormerStreamingCore(nn.Module):
    """CW-Former streaming core for ONNX export.

    Takes mel chunk + state tensors, returns log_probs + updated state.
    The mel frontend must be run separately in numpy.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.subsampling = model.subsampling
        self.encoder = model.encoder
        self.ctc_head = model.ctc_head
        self._config = model.config

    def forward(
        self,
        mel_chunk: torch.Tensor,
        pos_offset: torch.Tensor,
        kv_k_list: list[torch.Tensor],
        kv_v_list: list[torch.Tensor],
        conv_buf_list: list[torch.Tensor],
        sub_buf1: torch.Tensor,
        sub_buf2: torch.Tensor,
    ) -> tuple:
        # Conv subsampling (streaming)
        x, new_sub_buf1, new_sub_buf2 = self.subsampling.forward_streaming(
            mel_chunk, sub_buf1, sub_buf2,
        )

        T_out = x.shape[1]

        # Conformer encoder (streaming)
        kv_caches = [(kv_k_list[i], kv_v_list[i])
                     for i in range(len(kv_k_list))]
        offset = int(pos_offset[0].item())

        x, new_kv_caches, new_conv_buffers = self.encoder(
            x,
            kv_caches=kv_caches,
            conv_buffers=conv_buf_list,
            pos_offset=offset,
        )

        # CTC head
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, B, C)

        new_pos = pos_offset + T_out

        # Flatten outputs
        new_kv_k = [kv[0] for kv in new_kv_caches]
        new_kv_v = [kv[1] for kv in new_kv_caches]

        return (log_probs, new_pos, new_kv_k, new_kv_v,
                new_conv_buffers, new_sub_buf1, new_sub_buf2)


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def export_and_quantize(
    checkpoint: str,
    output_dir: str,
    opset: int = 17,
    benchmark_iters: int = 50,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    device = torch.device("cpu")
    model, model_cfg, sample_rate = _load_cwformer_checkpoint(
        checkpoint, device)

    n_mels = model_cfg.mel.n_mels
    n_params = sum(p.numel() for p in model.parameters())
    cfg = model_cfg.conformer
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_k = cfg.d_model // n_heads
    d_model = cfg.d_model
    conv_pad = cfg.conv_kernel - 1
    freq1 = math.ceil(n_mels / 2)  # freq dim after conv1

    print(f"Checkpoint : {checkpoint}")
    print(f"Parameters : {n_params:,}")
    print(f"Architecture: causal streaming Conformer")
    print(f"Mel config : n_mels={n_mels}, hop={model_cfg.mel.hop_length}, "
          f"f={model_cfg.mel.f_min}-{model_cfg.mel.f_max} Hz")
    print(f"State: {n_layers} layers x (KV cache + conv buffer) + 2 subsample buffers")

    # ---- Build exportable core ----
    core = _CWFormerStreamingCore(model)
    core.eval()

    # Dummy inputs: 1s of audio → ~100 mel frames, ~50 CTC frames after subsample
    T_mel = 100
    T_cached = 50  # some existing cache
    dummy_mel = torch.randn(1, T_mel, n_mels)
    dummy_pos = torch.tensor([T_cached], dtype=torch.long)

    dummy_kv_k = [torch.randn(1, n_heads, T_cached, d_k) for _ in range(n_layers)]
    dummy_kv_v = [torch.randn(1, n_heads, T_cached, d_k) for _ in range(n_layers)]
    dummy_conv_buf = [torch.randn(1, d_model, conv_pad) for _ in range(n_layers)]
    dummy_sub_buf1 = torch.randn(1, 1, 2, n_mels)
    dummy_sub_buf2 = torch.randn(1, model_cfg.subsample_channels, 2, freq1)

    # ---- Verify core runs ----
    print("\nVerifying streaming core ...")
    with torch.no_grad():
        out = core(dummy_mel, dummy_pos, dummy_kv_k, dummy_kv_v,
                   dummy_conv_buf, dummy_sub_buf1, dummy_sub_buf2)
    lp = out[0]
    print(f"  Output log_probs: {lp.shape} (expected ~50 CTC frames)")

    # ---- Build input/output names for ONNX ----
    input_names = ["mel_chunk", "pos_offset"]
    output_names = ["log_probs", "pos_offset_out"]
    dynamic_axes = {
        "mel_chunk": {1: "T_chunk"},
        "log_probs": {0: "T_out"},
    }

    dummy_inputs = [dummy_mel, dummy_pos]

    for i in range(n_layers):
        input_names.extend([f"kv_k_layer{i}", f"kv_v_layer{i}"])
        output_names.extend([f"kv_k_layer{i}_out", f"kv_v_layer{i}_out"])
        dynamic_axes[f"kv_k_layer{i}"] = {2: "T_cached"}
        dynamic_axes[f"kv_v_layer{i}"] = {2: "T_cached"}
        dynamic_axes[f"kv_k_layer{i}_out"] = {2: "T_cached_new"}
        dynamic_axes[f"kv_v_layer{i}_out"] = {2: "T_cached_new"}
        dummy_inputs.extend([dummy_kv_k[i], dummy_kv_v[i]])

    for i in range(n_layers):
        input_names.append(f"conv_buf_layer{i}")
        output_names.append(f"conv_buf_layer{i}_out")
        dummy_inputs.append(dummy_conv_buf[i])

    input_names.extend(["sub_buf1", "sub_buf2"])
    output_names.extend(["sub_buf1_out", "sub_buf2_out"])
    dummy_inputs.extend([dummy_sub_buf1, dummy_sub_buf2])

    # ---- Export FP32 ONNX ----
    fp32_path = str(out / "cwformer_streaming_fp32.onnx")
    print(f"\nExporting FP32 ONNX (opset {opset}) ...")
    print(f"  {len(input_names)} inputs, {len(output_names)} outputs")

    # We need a wrapper that takes flat args (ONNX doesn't support lists)
    class _FlatWrapper(nn.Module):
        def __init__(self, core, n_layers):
            super().__init__()
            self.core = core
            self.n_layers = n_layers

        def forward(self, *args):
            mel_chunk = args[0]
            pos_offset = args[1]
            idx = 2
            kv_k = list(args[idx:idx + self.n_layers])
            idx += self.n_layers
            kv_v = list(args[idx:idx + self.n_layers])
            idx += self.n_layers
            conv_bufs = list(args[idx:idx + self.n_layers])
            idx += self.n_layers
            sub_buf1 = args[idx]
            sub_buf2 = args[idx + 1]

            out = self.core(mel_chunk, pos_offset, kv_k, kv_v,
                            conv_bufs, sub_buf1, sub_buf2)

            # Flatten: log_probs, pos, kv_k_0..11, kv_v_0..11, conv_0..11, sub1, sub2
            result = [out[0], out[1]]
            for i in range(self.n_layers):
                result.append(out[2][i])  # kv_k
            for i in range(self.n_layers):
                result.append(out[3][i])  # kv_v
            for i in range(self.n_layers):
                result.append(out[4][i])  # conv_buf
            result.append(out[5])  # sub_buf1
            result.append(out[6])  # sub_buf2
            return tuple(result)

    flat_wrapper = _FlatWrapper(core, n_layers)
    flat_wrapper.eval()

    torch.onnx.export(
        flat_wrapper,
        tuple(dummy_inputs),
        fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    fp32_size = Path(fp32_path).stat().st_size / 1e6
    print(f"  Saved: {fp32_path} ({fp32_size:.1f} MB)")

    # ---- Verify FP32 ONNX ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[error] onnxruntime not installed. Install with:")
        print("  pip install onnxruntime")
        print("FP32 ONNX was saved but verification and INT8 skipped.")
        sys.exit(1)

    print("\nVerifying FP32 ONNX matches PyTorch ...")
    sess_fp32 = ort.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"])

    feed = {}
    for name, tensor in zip(input_names, dummy_inputs):
        feed[name] = tensor.numpy()

    ort_out = sess_fp32.run(None, feed)

    with torch.no_grad():
        pt_out = flat_wrapper(*dummy_inputs)
    max_diff = float(np.max(np.abs(pt_out[0].numpy() - ort_out[0])))
    print(f"  Max |diff| log-probs: {max_diff:.6f}")
    if max_diff > 0.01:
        print("  [warn] FP32 ONNX diverges from PyTorch — check export.")

    # ---- Quantize to INT8 ----
    print("\nQuantizing to INT8 (dynamic quantization) ...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("[error] onnxruntime.quantization not available.")
        sys.exit(1)

    int8_path = str(out / "cwformer_streaming_int8.onnx")
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
    )
    int8_size = Path(int8_path).stat().st_size / 1e6
    print(f"  Saved: {int8_path} ({int8_size:.1f} MB)")
    print(f"  Compression: {fp32_size:.1f} -> {int8_size:.1f} MB "
          f"({fp32_size / int8_size:.1f}x)")

    # ---- Benchmark ----
    print(f"\nBenchmarking ({benchmark_iters} iterations, "
          f"T={T_mel} mel frames ~ 1s audio) ...")

    sess_int8 = ort.InferenceSession(
        int8_path, providers=["CPUExecutionProvider"])

    # Warmup
    for _ in range(5):
        sess_fp32.run(None, feed)
        sess_int8.run(None, feed)

    # FP32
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        sess_fp32.run(None, feed)
    fp32_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    # INT8
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        sess_int8.run(None, feed)
    int8_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    # PyTorch FP32
    with torch.no_grad():
        for _ in range(5):
            flat_wrapper(*dummy_inputs)
        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            flat_wrapper(*dummy_inputs)
        pt_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    audio_duration_ms = T_mel * model_cfg.mel.hop_length / sample_rate * 1000
    print(f"  PyTorch FP32:     {pt_ms:7.1f} ms")
    print(f"  ONNX Runtime FP32:{fp32_ms:7.1f} ms")
    print(f"  ONNX Runtime INT8:{int8_ms:7.1f} ms")
    if int8_ms > 0:
        print(f"  INT8 speedup vs PyTorch: {pt_ms / int8_ms:.1f}x")

    rtf = int8_ms / audio_duration_ms
    print(f"  Real-time factor (INT8): {rtf:.3f}x "
          f"({'real-time OK' if rtf < 1.0 else 'too slow for real-time'})")

    # ---- Save mel config ----
    mel_config = {
        "sample_rate": sample_rate,
        "n_fft": model_cfg.mel.n_fft,
        "hop_length": model_cfg.mel.hop_length,
        "n_mels": n_mels,
        "f_min": model_cfg.mel.f_min,
        "f_max": model_cfg.mel.f_max,
        "architecture": "causal_streaming",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "d_k": d_k,
        "conv_kernel": cfg.conv_kernel,
        "max_cache_len": cfg.max_cache_len,
        "subsample_channels": model_cfg.subsample_channels,
    }
    config_path = str(out / "mel_config.json")
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    print(f"\nMel config: {config_path}")

    # ---- Summary ----
    ckpt_size = Path(checkpoint).stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"Original checkpoint: {ckpt_size:.1f} MB")
    print(f"FP32 ONNX model:     {fp32_size:.1f} MB")
    print(f"INT8 ONNX model:     {int8_size:.1f} MB")
    print(f"Per-chunk latency:   {int8_ms:.1f} ms / {audio_duration_ms:.0f}ms chunk (INT8)")
    print(f"Real-time factor:    {rtf:.3f}x")
    print(f"State tensors:       {2 * n_layers + n_layers + 2} "
          f"({n_layers} KV pairs + {n_layers} conv bufs + 2 subsample bufs)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export streaming CW-Former to INT8 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to CW-Former checkpoint (best_model.pt)")
    parser.add_argument("--output-dir", default="deploy", metavar="DIR",
                        dest="output_dir",
                        help="Directory for ONNX files and mel config")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--benchmark-iters", type=int, default=50,
                        dest="benchmark_iters",
                        help="Iterations for latency benchmark")

    args = parser.parse_args()
    export_and_quantize(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        opset=args.opset,
        benchmark_iters=args.benchmark_iters,
    )


if __name__ == "__main__":
    main()
