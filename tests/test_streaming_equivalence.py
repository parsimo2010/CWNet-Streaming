"""
Numerical equivalence test: model.forward() vs model.forward_streaming().

Feeds identical audio through the full-sequence training path and the
chunk-by-chunk streaming path, then compares log_probs element-wise.

If the two match to tolerance, the streaming code is correct and any
CER gap vs training-time validation comes from the input distribution
(e.g., SNR/length out of training range), not from the streaming mechanics.

If they diverge, the layer-by-layer diagnostics localize where.

The test runs under up to three configurations (when a GPU is
available; otherwise Runs 2 and 3 collapse into one):
  Run 1: primary device, default SDPA dispatch. On GPU the training
    path uses is_causal=True (Flash / mem-efficient kernel); the
    streaming path passes an explicit attn_mask and falls back to
    the math kernel. Different kernels => different fp reduction
    order; expected to FAIL atol=1e-3 on GPU even when correct.
  Run 2: primary device, SDPA forced to the math backend so both
    paths use the same kernel. Residual diff on GPU comes from
    cuDNN conv-algorithm variance (streaming chunks have different
    shapes than the full-sequence forward pass).
  Run 3: CPU-only. No cuDNN, single SDPA backend. This is the
    strictest signal of algorithmic correctness and matches the
    numerical regime of ONNX Runtime CPU inference, which is the
    deployment target.
Exit code is driven by Run 3 (or the forced-math run if CPU was
primary), and by any shape mismatch in any run.

Shape match is required: if the streaming path's tail length differs from
the full-forward path, that is a real bug (e.g., the `flush()` right-pad
is missing), not a harmless artifact — so the test treats any shape
mismatch as a failure rather than silently truncating to the shorter
length (which is what an earlier version of this test did, hiding the
mel tail asymmetry).

Run:
    python tests/test_streaming_equivalence.py --checkpoint checkpoints_full_penalty/best_model.pt
    python tests/test_streaming_equivalence.py --checkpoint checkpoints_full_penalty/best_model.pt --chunk-ms 500 --audio-sec 6 --wpm 25 --snr 10
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MorseConfig
from morse_generator import generate_sample
from neural_decoder.inference_cwformer import (
    CWFormerStreamingDecoder,
    _load_cwformer_checkpoint,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_CTX = True
except ImportError:
    # Older PyTorch: no torch.nn.attention module.
    _HAS_SDPA_CTX = False


def _math_sdpa_ctx():
    """Context that forces F.scaled_dot_product_attention onto the MATH
    backend, so the training path (is_causal=True) and streaming path
    (explicit attn_mask) use the *same* kernel and thus the same
    floating-point reduction order."""
    if _HAS_SDPA_CTX:
        return sdpa_kernel(SDPBackend.MATH)
    # Legacy fallback (PyTorch < 2.3).
    return torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_flash=False, enable_mem_efficient=False,
    )


def make_config(
    snr_db: float,
    wpm: float,
    min_chars: int = 20,
    max_chars: int = 30,
) -> MorseConfig:
    """Challenging-but-decodable CW audio at a fixed SNR and WPM. No
    augmentations, paddle key.

    Defaults are tuned for the equivalence test to land at the
    near-ambiguous edge of the model's operating envelope (slow WPM,
    low SNR), where small numerical diffs between full-forward and
    streaming have the best chance of flipping argmax at some frames.
    A clean high-SNR clip shows no decode difference even with
    substantial log_prob drift, because softmax is sharply peaked —
    exactly the scenario that *hides* divergence impact.
    """
    mc = MorseConfig()
    mc.min_snr_db = mc.max_snr_db = snr_db
    mc.min_wpm = mc.max_wpm = wpm
    mc.key_type_weights = (0.0, 0.0, 1.0, 0.0)  # paddle
    # Disable all augmentations — we want a deterministic, clean comparison.
    mc.agc_probability = 0.0
    mc.qsb_probability = 0.0
    mc.qrm_probability = 0.0
    mc.qrn_probability = 0.0
    mc.bandpass_probability = 0.0
    mc.hf_noise_probability = 0.0
    mc.farnsworth_probability = 0.0
    mc.multi_op_probability = 0.0
    mc.speed_drift_max = 0.0
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.0
    mc.min_chars = min_chars
    mc.max_chars = max_chars
    return mc


def _run_block_step_by_step(layer, x, kv_cache=None, conv_buffer=None, pos_offset=0):
    """Replicate ConformerBlock.forward but return intermediates at each sub-step."""
    sub = {}
    sub["input"] = x.clone()
    x = x + 0.5 * layer.ff1(x)
    sub["after_ff1"] = x.clone()
    mha_out, new_kv = layer.mha(x, kv_cache=kv_cache, pos_offset=pos_offset)
    sub["after_mha_out"] = mha_out.clone()
    x = x + mha_out
    sub["after_mha_residual"] = x.clone()
    conv_out, new_cb = layer.conv(x, conv_buffer=conv_buffer)
    sub["after_conv_out"] = conv_out.clone()
    x = x + conv_out
    sub["after_conv_residual"] = x.clone()
    x = x + 0.5 * layer.ff2(x)
    sub["after_ff2"] = x.clone()
    x = layer.final_norm(x)
    sub["after_final_norm"] = x.clone()
    return x, new_kv, new_cb, sub


def full_intermediates(model, audio: np.ndarray, device: torch.device) -> dict:
    """Run the training path and capture outputs at every major stage + every encoder block + sub-steps in block 0."""
    model.eval()
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    lengths = torch.tensor([audio_t.shape[1]], device=device)
    out = {}
    with torch.no_grad():
        mel, mel_lens = model.mel_frontend(audio_t, lengths)
        out["mel"] = mel

        subsample_out, sub_lens = model.subsampling(mel, mel_lens)
        out["after_subsample"] = subsample_out

        # Step through encoder block-by-block to capture intermediates
        x = subsample_out
        for i, layer in enumerate(model.encoder.layers):
            if i == 0:
                # Detailed sub-step capture for block 0
                x, _, _, sub = _run_block_step_by_step(layer, x)
                for name, tensor in sub.items():
                    out[f"block0_{name}"] = tensor
            else:
                x, _, _ = layer(x)
            out[f"after_block_{i}"] = x.clone()
        enc_out = x
        out["after_encoder"] = enc_out

        logits = model.ctc_head(enc_out)
        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        out["log_probs"] = log_probs
    return out


def streaming_intermediates(
    model, audio: np.ndarray, chunk_samples: int, device: torch.device
) -> dict:
    """Run the streaming path and concatenate outputs at matching stages,
    including block-by-block outputs from the encoder.

    The final chunk is right-padded by ``n_fft // 2`` zeros to mirror
    ``CWFormerStreamingDecoder.flush()``, so the streaming tail frame
    count matches the full-forward path (which pads both sides of the
    full utterance). Without this pad, the streaming path is short by
    1–2 frames at the tail.
    """
    model.eval()
    state = model.init_streaming_state(device)
    n_layers = len(model.encoder.layers)
    mels = []
    subsample_outs = []
    block_outs: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    block0_subs: dict[str, list[torch.Tensor]] = {}
    log_probs_chunks = []

    n_fft = model.config.mel.n_fft
    pad_right = n_fft // 2

    # Iterate over chunk_samples-sized slices of the raw audio; on the
    # FINAL chunk, right-pad by n_fft//2 zeros to mirror
    # CWFormerStreamingDecoder.flush(). This matches forward()'s
    # both-sides pad in MelFrontend and is what makes the streaming
    # tail frame count line up with the full-forward path.
    pos = 0
    with torch.no_grad():
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunk_np = audio[pos:end]
            # If this is the final chunk, right-pad it by n_fft//2.
            is_final = end == len(audio)
            if is_final:
                chunk_np = np.concatenate(
                    [chunk_np, np.zeros(pad_right, dtype=chunk_np.dtype)]
                )
            chunk = torch.from_numpy(chunk_np).unsqueeze(0).to(device)

            # Stage 1: mel
            mel, new_stft = model.mel_frontend.compute_streaming(
                chunk, state.get("stft_buffer")
            )
            state["stft_buffer"] = new_stft
            if mel.shape[1] > 0:
                mels.append(mel)

            # Stage 2: subsampling
            if mel.shape[1] > 0:
                x, nb1, nb2 = model.subsampling.forward_streaming(
                    mel, state["sub_buf1"], state["sub_buf2"],
                )
                state["sub_buf1"] = nb1
                state["sub_buf2"] = nb2
                if x.shape[1] > 0:
                    subsample_outs.append(x)

                    # Stage 3: encoder, block-by-block
                    pos_offset = state["pos_offset"]
                    new_kv_caches = []
                    new_conv_buffers = []
                    max_cache = model.config.conformer.max_cache_len
                    h = x
                    for i, layer in enumerate(model.encoder.layers):
                        kv_i = state["kv_caches"][i]
                        cb_i = state["conv_buffers"][i]
                        if i == 0:
                            h, new_kv, new_cb, sub = _run_block_step_by_step(
                                layer, h, kv_cache=kv_i, conv_buffer=cb_i,
                                pos_offset=pos_offset,
                            )
                            for name, tensor in sub.items():
                                block0_subs.setdefault(name, []).append(tensor)
                        else:
                            h, new_kv, new_cb = layer(
                                h, kv_cache=kv_i, conv_buffer=cb_i,
                                pos_offset=pos_offset,
                            )
                        block_outs[i].append(h.clone())
                        # Trim KV cache
                        k_c, v_c = new_kv
                        if k_c.shape[2] > max_cache:
                            k_c = k_c[:, :, -max_cache:, :]
                            v_c = v_c[:, :, -max_cache:, :]
                        new_kv_caches.append((k_c, v_c))
                        new_conv_buffers.append(new_cb)
                    state["kv_caches"] = new_kv_caches
                    state["conv_buffers"] = new_conv_buffers
                    state["pos_offset"] += x.shape[1]

                    # Stage 4: CTC head
                    import torch.nn.functional as F
                    logits = model.ctc_head(h)
                    lp = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    log_probs_chunks.append(lp)

            pos = end

    cat_mel = torch.cat(mels, dim=1) if mels else torch.zeros(1, 0, model.config.mel.n_mels, device=device)
    cat_sub = torch.cat(subsample_outs, dim=1) if subsample_outs else torch.zeros(1, 0, model.config.conformer.d_model, device=device)
    cat_blocks = {
        f"after_block_{i}": torch.cat(block_outs[i], dim=1)
        if block_outs[i] else torch.zeros(1, 0, model.config.conformer.d_model, device=device)
        for i in range(n_layers)
    }
    cat_enc = cat_blocks[f"after_block_{n_layers - 1}"]
    cat_lp = torch.cat(log_probs_chunks, dim=0) if log_probs_chunks else torch.zeros(0, 1, model.config.num_classes, device=device)

    result = {
        "mel": cat_mel,
        "after_subsample": cat_sub,
        "after_encoder": cat_enc,
        "log_probs": cat_lp,
    }
    result.update(cat_blocks)
    # Concatenate block 0 sub-step captures across chunks
    for name, tensors in block0_subs.items():
        if tensors:
            result[f"block0_{name}"] = torch.cat(tensors, dim=1)
    return result


def summarize_diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    """Print element-wise comparison of two tensors of the same shape."""
    if a.shape != b.shape:
        print(f"  {label}: SHAPE MISMATCH  full={tuple(a.shape)} streaming={tuple(b.shape)}")
        return
    a = a.float()
    b = b.float()
    abs_diff = (a - b).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rel = abs_diff / (a.abs().clamp(min=1e-6))
    max_rel = rel.max().item()
    print(f"  {label}: shape={tuple(a.shape)}  max_abs_diff={max_abs:.4e}  "
          f"mean_abs_diff={mean_abs:.4e}  max_rel_diff={max_rel:.4e}")


def main():
    parser = argparse.ArgumentParser(description="Streaming vs full-forward equivalence test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-ms", type=int, default=500, dest="chunk_ms")
    parser.add_argument("--wpm", type=float, default=15.0,
                        help="WPM (default 15 — slow end of training "
                             "envelope; surfaces near-ambiguous decodes)")
    parser.add_argument("--snr", type=float, default=3.0,
                        help="SNR in dB (default 3 — low end of full "
                             "curriculum; high-SNR clips hide divergence "
                             "because softmax is too sharp for small "
                             "log_prob diffs to flip argmax)")
    parser.add_argument("--min-chars", type=int, default=20, dest="min_chars")
    parser.add_argument("--max-chars", type=int, default=30, dest="max_chars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for log_prob comparison")
    parser.add_argument("--skip-cpu-run", action="store_true",
                        help="Skip the CPU-only run (faster on GPU hosts "
                             "where you just want the two GPU runs)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model on {device}...")
    model, model_cfg, sample_rate = _load_cwformer_checkpoint(args.checkpoint, device)
    # Do NOT override max_cache_len. The streaming path trims the KV cache
    # to this many frames; full-forward never trims. If test audio produces
    # more subsample frames than max_cache_len, the trimmed keys are
    # invisible to late queries in streaming but still seen by full-forward
    # causal attention — that's a real algorithmic difference, and it makes
    # this test report huge MHA diffs that look like a bug but are just the
    # cap doing its job. Use whatever the checkpoint shipped with (the
    # ConformerConfig default is 1475 ≈ 29.5 s at 50 fps, enough for any
    # in-distribution test audio up to 30 s).
    #
    # We DO assert here that the test audio won't exceed the cap, so a
    # misconfigured long-audio test fails loudly instead of silently
    # producing the "truncated past" divergence.
    max_cache_frames = model.config.conformer.max_cache_len
    frames_per_sec = sample_rate // model_cfg.mel.hop_length // 2
    print(
        f"  max_cache_len={max_cache_frames} frames "
        f"(~{max_cache_frames / frames_per_sec:.1f}s of audio @ {frames_per_sec} fps)"
    )

    print(f"Generating test audio (WPM={args.wpm}, SNR={args.snr}, "
          f"chars=[{args.min_chars},{args.max_chars}], paddle, no aug)...")
    mc = make_config(args.snr, args.wpm, args.min_chars, args.max_chars)
    rng = np.random.default_rng(args.seed)
    audio, text, meta = generate_sample(mc, rng=rng)
    print(f"  text: {text!r}")
    print(f"  audio: {len(audio)} samples = {len(audio)/sample_rate:.2f}s")

    chunk_samples = int(args.chunk_ms * sample_rate / 1000)
    print(f"  chunk_samples: {chunk_samples}  ({args.chunk_ms} ms)")

    # Bail out if the audio would force the streaming path to trim its
    # KV cache. Streaming trims; full-forward doesn't. Any audio that
    # produces more subsample frames than max_cache_len will make the
    # two paths disagree at positions past the cap — not because the
    # streaming code is wrong, but because the cap is doing its job.
    expected_sub_frames = len(audio) // (model_cfg.mel.hop_length * 2) + 1
    if expected_sub_frames > max_cache_frames:
        raise SystemExit(
            f"Test audio ({len(audio)/sample_rate:.2f}s, ~{expected_sub_frames} "
            f"subsample frames) exceeds max_cache_len={max_cache_frames}. "
            f"The streaming path will trim the KV cache; full-forward won't. "
            f"Shorten the audio (reduce --wpm's min_chars floor, or set a "
            f"shorter clip) or raise max_cache_len in the checkpoint config."
        )

    # --- Run both paths under three configurations:
    #   1) primary device, default SDPA dispatch (on GPU: different
    #      kernels per path; on CPU: only the math kernel exists).
    #   2) primary device, SDPA forced to MATH backend.
    #   3) CPU-only (skipped if primary device is already CPU). This
    #      is the strictest check because CPU eliminates cuDNN
    #      algorithm variance in the conv subsampling AND has only
    #      the math SDPA backend. It also mirrors ONNX Runtime CPU
    #      inference, which is where the deployed model ends up.
    cpu_device = torch.device("cpu")
    run_configs = [
        (
            "RUN 1: primary device, default SDPA dispatch",
            device,
            contextlib.nullcontext(),
            (
                "On CUDA: training uses is_causal=True -> Flash / "
                "mem-efficient kernel; streaming uses explicit attn_mask "
                "-> math kernel. Different reduction order => fp noise "
                "that amplifies through 12 layers; expected to FAIL "
                "atol=1e-3 on log_probs while still producing identical "
                "decodes. On CPU: only math backend is available, so "
                "this equals RUN 2."
            ),
        ),
        (
            "RUN 2: primary device, SDPA forced to MATH backend",
            device,
            _math_sdpa_ctx(),
            (
                "Both paths dispatch to the math kernel, so SDPA "
                "reductions agree. Residual diff comes from cuDNN "
                "algorithm variance in the conv subsampling (streaming "
                "chunks have different shapes than the full-forward "
                "pass). On GPU this typically lands in the 1e-3 to "
                "1e-2 range after 12 layers; on CPU it collapses to "
                "~1e-5."
            ),
        ),
    ]
    if device.type != "cpu" and not args.skip_cpu_run:
        run_configs.append((
            "RUN 3: CPU-only (no cuDNN, single SDPA backend)",
            cpu_device,
            contextlib.nullcontext(),
            (
                "Move the model to CPU and run both paths there. CPU "
                "uses deterministic reference kernels for conv and has "
                "only the math SDPA backend, so residual diffs are "
                "purely from the chunk-vs-full-sequence processing "
                "(not from kernel dispatch or cuDNN heuristics). This "
                "matches the numerical regime of ONNX Runtime CPU "
                "inference, which is the deployment target."
            ),
        ))

    results: list[tuple[str, bool, bool]] = []  # (label, ok, any_shape_mismatch)

    for label, run_device, ctx, desc in run_configs:
        print("\n" + "=" * 72)
        print(f"  {label}")
        print("=" * 72)
        print(f"  {desc}")

        # Move model to the run's device if needed. RoPE cos/sin
        # buffers are non-persistent and ride along with .to(); the
        # streaming state is rebuilt per-run on run_device below.
        current_device = next(model.parameters()).device
        if current_device != run_device:
            print(f"  (moving model: {current_device} -> {run_device})")
            model.to(run_device)

        with ctx:
            print("\nFull forward (captures intermediates at every stage)...")
            full = full_intermediates(model, audio, run_device)
            print("Streaming forward (captures intermediates at every stage)...")
            stream = streaming_intermediates(model, audio, chunk_samples, run_device)

        # Track whether any stage has mismatched shapes — a length mismatch
        # at the tail is the symptom of the streaming/forward asymmetry
        # fixed by the `flush()` right-pad. Any mismatch here is a hard
        # failure rather than a silently-truncated comparison.
        any_shape_mismatch = [False]

        print("\nLayer-by-layer comparison (requires matching shapes):")
        def compare(label, tf, ts, time_dim, _sm=any_shape_mismatch):
            nf = tf.shape[time_dim]
            ns = ts.shape[time_dim]
            if nf != ns:
                _sm[0] = True
                print(
                    f"  {label}: SHAPE MISMATCH  full T={nf}  stream T={ns}  "
                    f"(diff={nf - ns})  -- treating as failure; comparison "
                    f"skipped to avoid silent truncation"
                )
                return
            summarize_diff(tf, ts, label)

        compare("mel            ", full["mel"],             stream["mel"],             time_dim=1)
        compare("after_subsample", full["after_subsample"], stream["after_subsample"], time_dim=1)
        # Sub-step detail inside block 0 (where the big amplification happens)
        block0_order = [
            "block0_input", "block0_after_ff1", "block0_after_mha_out",
            "block0_after_mha_residual", "block0_after_conv_out",
            "block0_after_conv_residual", "block0_after_ff2",
            "block0_after_final_norm",
        ]
        for key in block0_order:
            if key in full and key in stream:
                compare(f"{key:<25s}", full[key], stream[key], time_dim=1)
        # Block-by-block inside the encoder
        n_layers = sum(1 for k in full.keys() if k.startswith("after_block_"))
        for i in range(n_layers):
            key = f"after_block_{i}"
            compare(f"{key:<25s}", full[key], stream[key], time_dim=1)
        compare("log_probs                ", full["log_probs"], stream["log_probs"], time_dim=0)

        # Decode both paths and compare text output
        from neural_decoder.train_cwformer import greedy_decode, compute_cer
        hyp_full = greedy_decode(full["log_probs"][:, 0, :])
        hyp_stream = greedy_decode(stream["log_probs"][:, 0, :]) if stream["log_probs"].shape[0] > 0 else ""
        cer_full = compute_cer(hyp_full, text)
        cer_stream = compute_cer(hyp_stream, text)

        print(f"\nDecode comparison:")
        print(f"  ref:       {text!r}")
        print(f"  full:      {hyp_full!r}  (CER={cer_full:.1%})")
        print(f"  streaming: {hyp_stream!r}  (CER={cer_stream:.1%})")

        T_full = full["log_probs"].shape[0]
        T_stream = stream["log_probs"].shape[0]

        if T_full != T_stream:
            any_shape_mismatch[0] = True
            print(
                f"\nlog_probs SHAPE MISMATCH: full T={T_full} stream T={T_stream}. "
                f"This must match exactly -- the streaming flush() right-pad is "
                f"what keeps the tail aligned with forward()'s both-sides pad."
            )
            ok = False
        else:
            ok = torch.allclose(
                full["log_probs"], stream["log_probs"], atol=args.atol,
            )

        print(
            f"\nallclose(atol={args.atol:.0e}) on log_probs "
            f"(requires matching shape): {'PASS' if ok and not any_shape_mismatch[0] else 'FAIL'}"
        )
        if any_shape_mismatch[0]:
            print("  ->At least one stage reported a shape mismatch. The streaming")
            print("    path is not aligned with the full-forward path; this must be")
            print("    fixed before numerical tolerances are meaningful.")
        elif not ok:
            print("  ->Look at the layer-by-layer table above. The first stage that")
            print("    shows a large max_abs_diff is where the bug lives.")
        else:
            print("  ->Streaming is numerically equivalent to full-forward.")

        results.append((label, ok, any_shape_mismatch[0]))

    # --- Summary across all runs ---
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for label, ok, shape_mismatch in results:
        status = "PASS" if ok and not shape_mismatch else "FAIL"
        print(f"  [{status}] {label}")
    print(
        "\n  Run 1 is expected to FAIL on GPU at atol=1e-3 (Flash vs math "
        "\n  kernel mismatch). Run 2 (forced math backend) usually also "
        "\n  FAILs on GPU due to cuDNN conv algorithm variance — that's "
        "\n  fp noise, not a bug. The strictest signal is Run 3 (CPU-only): "
        "\n  no cuDNN, single SDPA backend, deterministic reductions. If "
        "\n  Run 3 PASSes and decodes match on all runs, the streaming "
        "\n  code is algorithmically correct and the deployment path "
        "\n  (ONNX CPU) will behave the same as full-forward."
    )

    # Exit non-zero ONLY on:
    #   - shape mismatch in any run (real bug, flush() pad / subsample
    #     buffer misalignment, etc.)
    #   - failure of the strictest run (CPU-only if present, else the
    #     forced-math run). GPU-only failures at 1e-3 are expected fp
    #     noise, not a correctness issue.
    any_shape_bug = any(sm for _, _, sm in results)
    strictest_ok = results[-1][1]  # last run is the strictest
    sys.exit(0 if strictest_ok and not any_shape_bug else 1)


if __name__ == "__main__":
    main()
