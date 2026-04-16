#!/usr/bin/env python3
"""
train_cwformer.py — Training loop for the causal streaming CW-Former.

The model uses fully causal attention (is_causal=True) and causal
convolutions during training. No explicit mask construction is needed —
causality is enforced internally by the model architecture. Weights are
shape-compatible with the original bidirectional CW-Former for fine-tuning.

Usage:
    # Quick test (verify pipeline)
    python -m neural_decoder.train_cwformer --scenario test

    # Stage 1: Clean conditions
    python -m neural_decoder.train_cwformer --scenario clean

    # Stage 2: Resume from clean, moderate augmentations
    python -m neural_decoder.train_cwformer --scenario moderate \
        --checkpoint checkpoints_cwformer/best_model.pt

    # Stage 3: Resume from moderate, full augmentations
    python -m neural_decoder.train_cwformer --scenario full \
        --checkpoint checkpoints_cwformer/best_model_moderate.pt

    # Fine-tune from bidirectional CWNet checkpoint
    python -m neural_decoder.train_cwformer --scenario full \
        --checkpoint /path/to/cwnet_bidirectional/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, create_default_config
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.conformer import ConformerConfig
from neural_decoder.mel_frontend import MelFrontendConfig
from neural_decoder.dataset_audio import AudioDataset, collate_fn


# ---------------------------------------------------------------------------
# CTC decoding helpers
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor) -> str:
    return vocab_module.decode_ctc(log_probs, blank_idx=0, strip_trailing_space=True)


def beam_decode(log_probs: torch.Tensor, beam_width: int = 10) -> str:
    return vocab_module.beam_search_ctc(
        log_probs, beam_width=beam_width, blank_idx=0, strip_trailing_space=True
    )


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    # Strip boundary spaces — the model is trained with [space]+text+[space]
    # targets but the reference text does not include boundary tokens.
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


# ---------------------------------------------------------------------------
# Buffer pre-generation
# ---------------------------------------------------------------------------

def generate_disk_cache(
    dataset: "AudioDataset",
    micro_batch: int,
    num_workers: int,
    buffer_epochs: int = 1,
    cache_dir: str = "",
    buffer_gen: int = 0,
) -> list:
    """Pre-generate buffer_epochs × epoch_size samples and save to disk.

    Returns a list of Path objects (file paths to .pt batch files), NOT
    loaded tensors.  The training loop loads batches lazily during iteration
    to keep RAM usage constant regardless of buffer size.
    """
    loader = DataLoader(
        dataset, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    index_file = cache_path / f"gen{buffer_gen}_index.pt"

    if index_file.exists():
        file_list = torch.load(index_file, weights_only=False)
        print(f"  Reusing cached buffer: {len(file_list)} batches from {cache_dir}",
              file=sys.stderr)
        return file_list

    file_list = []
    batch_idx = 0
    for pass_idx in range(buffer_epochs):
        desc = (f"Caching buffer pass {pass_idx + 1}/{buffer_epochs}"
                if buffer_epochs > 1 else "Caching buffer")
        for batch in tqdm(loader, desc=desc, file=sys.stderr, leave=False):
            p = cache_path / f"gen{buffer_gen}_batch{batch_idx:06d}.pt"
            torch.save(batch, p)
            file_list.append(p)
            batch_idx += 1
    torch.save(file_list, index_file)
    return file_list


def _lazy_disk_iter(file_paths: list):
    """Yield batches by loading one .pt file at a time from disk."""
    for p in file_paths:
        yield torch.load(p, weights_only=False)


def generate_epoch_buffer(
    dataset: "AudioDataset",
    micro_batch: int,
    num_workers: int,
    buffer_epochs: int = 1,
) -> list:
    """Pre-generate buffer_epochs × epoch_size samples into a list of batches (in-memory)."""
    loader = DataLoader(
        dataset, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )

    buffer = []
    for pass_idx in range(buffer_epochs):
        desc = (f"Filling buffer pass {pass_idx + 1}/{buffer_epochs}"
                if buffer_epochs > 1 else "Filling buffer")
        for batch in tqdm(loader, desc=desc, file=sys.stderr, leave=False):
            buffer.append(batch)
    return buffer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: CWFormer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    beam_width: int = 0,
) -> dict:
    """Evaluate model on a dataset. Returns dict with loss, greedy_cer, beam_cer."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_batches = 0
    all_cer_greedy = []
    all_cer_beam = []

    with torch.no_grad():
        for audio, targets, audio_lens, target_lens, texts in loader:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)

                # CTC feasibility: skip samples where output is too short
                valid = out_lens >= target_lens
                if not valid.all():
                    idx = valid.nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        del audio, targets, audio_lens, target_lens, log_probs, out_lens
                        continue
                    log_probs = log_probs[:, idx, :]
                    targets = targets[idx]
                    out_lens = out_lens[idx]
                    target_lens = target_lens[idx]
                    texts = [texts[i] for i in idx.cpu().tolist()]

                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            total_loss += loss.item()
            total_batches += 1

            # Move to CPU for CER computation, free GPU memory
            log_probs_cpu = log_probs.cpu()
            out_lens_cpu = out_lens.cpu()
            del audio, targets, audio_lens, target_lens, log_probs, out_lens, loss

            B = log_probs_cpu.shape[1]
            for i in range(B):
                T_i = int(out_lens_cpu[i].item())
                lp_i = log_probs_cpu[:T_i, i, :]

                hyp_greedy = greedy_decode(lp_i)
                cer_g = compute_cer(hyp_greedy, texts[i])
                all_cer_greedy.append(cer_g)

                if beam_width > 0:
                    hyp_beam = beam_decode(lp_i, beam_width)
                    cer_b = compute_cer(hyp_beam, texts[i])
                    all_cer_beam.append(cer_b)

            del log_probs_cpu, out_lens_cpu

    results = {
        "loss": total_loss / max(1, total_batches),
        "greedy_cer": float(np.mean(all_cer_greedy)) if all_cer_greedy else 1.0,
    }
    if all_cer_beam:
        results["beam_cer"] = float(np.mean(all_cer_beam))
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if args.no_amp or is_rocm:
        use_amp = False
    else:
        use_amp = device.type == "cuda"
    # pin_memory is counter-productive on unified memory (e.g. AMD APU / iGPU)
    use_pin_memory = device.type == "cuda" and not is_rocm
    if is_rocm:
        print(f"Device: {device} (ROCm {torch.version.hip}), AMP disabled, pin_memory disabled")
    else:
        print(f"Device: {device}, AMP: {use_amp}")

    if device.type == "cuda":
        # Auto-tune convolution algorithms for consistent input sizes
        torch.backends.cudnn.benchmark = True
        # Enable TF32 on Ampere+ for faster fp32 matmuls via tensor cores
        if not is_rocm:
            torch.set_float32_matmul_precision('high')

    # ---- Config ----
    config = create_default_config(args.scenario)

    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    # ---- Model config ----
    mel_cfg = MelFrontendConfig(
        sample_rate=config.morse.sample_rate,
        spec_augment=True,
    )
    conformer_cfg = ConformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    )
    model_cfg = CWFormerConfig(
        mel=mel_cfg,
        conformer=conformer_cfg,
    )

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = ckpt_dir / "config.json"
    config.save(str(config_path))

    # ---- Model ----
    model = CWFormer(model_cfg).to(device)
    print(f"CW-Former (causal streaming): {model.num_params:,} parameters")
    print(f"  d_model={conformer_cfg.d_model}, n_heads={conformer_cfg.n_heads}, "
          f"n_layers={conformer_cfg.n_layers}, d_ff={conformer_cfg.d_ff}, "
          f"conv_kernel={conformer_cfg.conv_kernel}")
    print(f"  subsample=2x (20ms/frame), causal attention + causal conv")

    # ---- Load checkpoint if resuming ----
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt = None
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        prev_scenario = ckpt.get("scenario", "")
        if prev_scenario == args.scenario and "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        else:
            best_val_loss = float("inf")
            if prev_scenario and prev_scenario != args.scenario:
                print(f"  Scenario changed ({prev_scenario} -> {args.scenario}), resetting best_val_loss")
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    total_epochs = config.training.num_epochs

    if args.lr_resume and ckpt is not None:
        # ---- Continue LR schedule from checkpoint ----
        # Restore optimizer state first (momentum buffers + saved LR)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        warmup_epochs = min(5, max(1, total_epochs // 40))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Restore scheduler position on the cosine curve
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        print(f"  LR resumed at {optimizer.param_groups[0]['lr']:.2e} "
              f"(epoch {start_epoch}/{total_epochs})")
    else:
        # ---- Fresh LR schedule over remaining epochs ----
        # Restore optimizer momentum buffers but reset LR to args.lr
        if ckpt is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
                pg.pop("initial_lr", None)

        remaining_epochs = total_epochs - start_epoch
        warmup_epochs = min(5, max(1, remaining_epochs // 40))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, remaining_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if start_epoch > 0:
            print(f"  Fresh LR schedule: {remaining_epochs} remaining epochs, "
                  f"peak lr={args.lr:.2e}")

    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Loss ----
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # ---- Datasets ----
    # Audio generation is CPU-bound — use smaller batches and fewer samples
    samples_per_epoch = min(config.training.samples_per_epoch, 20000)
    val_samples = min(config.training.val_samples, 2000)

    micro_batch = args.batch_size
    effective_batch = 64  # Target effective batch for audio
    accum_steps = max(1, effective_batch // micro_batch)

    train_ds = AudioDataset(
        config, epoch_size=samples_per_epoch, seed=None,
        qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
    )
    val_ds = AudioDataset(
        config, epoch_size=val_samples, seed=None,  # fresh samples each eval
        qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
    )

    num_workers = args.workers
    reuse_factor = args.reuse_factor

    # For reuse_factor==1 keep the persistent streaming loader (current behaviour).
    # For reuse_factor>1 we generate batches into RAM and replay them.
    if reuse_factor <= 1:
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=use_pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=min(num_workers, 4), pin_memory=use_pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=min(num_workers, 4) > 0,
    )

    # ---- CSV log ----
    log_path = ckpt_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "greedy_cer", "beam_cer", "lr", "time_s"]
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    buffer_epochs = args.buffer_epochs
    cache_dir = args.cache_dir

    # ---- Training loop ----
    beam_cer_interval = 50
    reuse_str = (f", buffer_epochs={buffer_epochs}, reuse_factor={reuse_factor}"
                 if reuse_factor > 1 else "")
    print(f"\nTraining: {total_epochs} epochs, {samples_per_epoch} samples/epoch, "
          f"micro_batch={micro_batch}, accum={accum_steps} (effective={micro_batch*accum_steps}), "
          f"workers={num_workers}{reuse_str}")
    print(f"Scenario: {args.scenario}"
          + (f", cache_dir={cache_dir}" if cache_dir else ""))

    # Buffer state for reuse_factor > 1.
    #
    # In-memory path (cache_dir=None):
    #   Fill phase: generate one epoch at a time, train on fresh data only.
    #   Replay does not start until buffer_epochs fill epochs are done.
    #   Replay phase: shuffle full buffer, slice to one epoch's worth of batches.
    #
    # Disk-cache path (cache_dir set):
    #   Blocking fill: generates all buffer_epochs passes up front, saves to disk.
    #   GPU is idle during fill but disk enables buffers too large for RAM.
    if reuse_factor > 1 and reuse_factor <= buffer_epochs:
        print(f"WARNING: reuse_factor ({reuse_factor}) <= buffer_epochs "
              f"({buffer_epochs}). No replay will occur. "
              f"Increase --reuse-factor above --buffer-epochs.", file=sys.stderr)
    _buffer: list = []
    _buffer_rng = np.random.default_rng(99)
    # In-memory state
    _phase: str = "fill"
    _fill_count: int = 0
    _replay_count: int = 0
    _batches_per_epoch: int = 0
    # Disk-cache state
    _buffer_gen: int = -1

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        if reuse_factor > 1:
            if cache_dir is not None:
                # ---- Disk-cache path: lazy loading ----
                # Batches stay on disk; loaded one at a time during iteration.
                # RAM usage is O(1) regardless of buffer size.
                this_gen = epoch // reuse_factor
                if this_gen != _buffer_gen:
                    if _buffer_gen >= 0:
                        prev_index = Path(cache_dir) / f"gen{_buffer_gen}_index.pt"
                        if prev_index.exists():
                            prev_files = torch.load(prev_index, weights_only=False)
                            for p in prev_files:
                                try:
                                    Path(p).unlink()
                                except OSError:
                                    pass
                            prev_index.unlink(missing_ok=True)
                    _buffer_gen = this_gen
                    t_buf = time.time()
                    print(f"\nFilling disk buffer (gen {_buffer_gen + 1}, "
                          f"epoch {epoch + 1}, "
                          f"{buffer_epochs} pass(es))...", file=sys.stderr)
                    _buffer = generate_disk_cache(
                        train_ds, micro_batch, num_workers,
                        buffer_epochs=buffer_epochs,
                        cache_dir=cache_dir,
                        buffer_gen=_buffer_gen,
                    )
                    print(f"  {len(_buffer)} batches "
                          f"({len(_buffer) * micro_batch:,} samples) "
                          f"in {time.time() - t_buf:.0f}s", file=sys.stderr)
                # Shuffle file paths, then load lazily during iteration
                shuffled = list(_buffer)
                _buffer_rng.shuffle(shuffled)
                train_iter = _lazy_disk_iter(shuffled)
                pbar_total = len(shuffled)
            else:
                # ---- In-memory path: fill-as-you-go ----
                if _phase == "fill":
                    t_buf = time.time()
                    print(f"\nFill {_fill_count + 1}/{buffer_epochs} "
                          f"(epoch {epoch + 1})...", file=sys.stderr)
                    new_batches = generate_epoch_buffer(
                        train_ds, micro_batch, num_workers, 1)
                    _buffer.extend(new_batches)
                    if _batches_per_epoch == 0:
                        _batches_per_epoch = len(new_batches)
                    _fill_count += 1
                    print(f"  {len(new_batches)} batches in {time.time() - t_buf:.0f}s "
                          f"(buffer: {len(_buffer)} total, "
                          f"{_fill_count}/{buffer_epochs} passes).", file=sys.stderr)
                    # Train on freshly generated data only — replay not yet started.
                    shuffled_new = list(new_batches)
                    _buffer_rng.shuffle(shuffled_new)
                    train_iter = iter(shuffled_new)
                    pbar_total = len(shuffled_new)
                    if _fill_count >= buffer_epochs:
                        _phase = "replay"
                else:
                    # Replay: shuffle full buffer, slice to one epoch's worth.
                    shuffled = list(_buffer)
                    _buffer_rng.shuffle(shuffled)
                    train_iter = iter(shuffled[:_batches_per_epoch])
                    pbar_total = _batches_per_epoch
                    _replay_count += 1
                    if _replay_count >= reuse_factor - buffer_epochs:
                        _buffer = []
                        _fill_count = 0
                        _replay_count = 0
                        _batches_per_epoch = 0
                        _phase = "fill"
        else:
            train_iter = iter(train_loader)
            pbar_total = None

        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{total_epochs}",
                     leave=False, file=sys.stderr, total=pbar_total)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        running_loss = torch.tensor(0.0, device=device)

        for audio, targets, audio_lens, target_lens, texts in pbar:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)

                # CTC feasibility: clamp output lengths to be at least as
                # long as targets.  zero_infinity=True handles any remaining
                # infeasible paths without needing a GPU-syncing .all() check.
                out_lens = out_lens.clamp(min=1)

                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)
                loss = loss / accum_steps

            # No isnan/isinf check — GradScaler already skips optimizer
            # steps when gradients contain inf/nan.  Checking here would
            # force a CPU-GPU sync on every micro-step.

            scaler.scale(loss).backward()

            # Accumulate loss on GPU to avoid .item() sync every micro-step
            running_loss += loss.detach() * accum_steps
            del audio, targets, audio_lens, target_lens, log_probs, out_lens, loss

            micro_step += 1

            if micro_step >= accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            n_batches += 1

        # Flush any remaining gradient
        if micro_step > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Single GPU→CPU sync per epoch for loss logging (not per micro-step)
        avg_train_loss = running_loss.item() / max(1, n_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        if is_rocm:
            torch.cuda.empty_cache()
        do_beam = (epoch + 1) % beam_cer_interval == 0 or epoch == total_epochs - 1
        val_results = evaluate(
            model, val_loader, device, use_amp,
            beam_width=10 if do_beam else 0,
        )

        elapsed = time.time() - t0
        val_loss = val_results["loss"]
        greedy_cer = val_results["greedy_cer"]
        beam_cer = val_results.get("beam_cer", -1.0)

        print(f"Epoch {epoch+1:4d}/{total_epochs} | "
              f"train={avg_train_loss:.4f} val={val_loss:.4f} | "
              f"CER={greedy_cer:.3f}"
              + (f" beam={beam_cer:.3f}" if beam_cer >= 0 else "")
              + f" | lr={current_lr:.2e} | {elapsed:.0f}s")

        # ---- CSV log ----
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{avg_train_loss:.6f}", f"{val_loss:.6f}",
                f"{greedy_cer:.6f}", f"{beam_cer:.6f}" if beam_cer >= 0 else "",
                f"{current_lr:.2e}", f"{elapsed:.1f}",
            ])

        # ---- Checkpoints ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "greedy_cer": greedy_cer,
            "scenario": args.scenario,
            "total_epochs": total_epochs,
            "model_config": {
                "d_model": conformer_cfg.d_model,
                "n_heads": conformer_cfg.n_heads,
                "n_layers": conformer_cfg.n_layers,
                "d_ff": conformer_cfg.d_ff,
                "conv_kernel": conformer_cfg.conv_kernel,
                "max_cache_len": conformer_cfg.max_cache_len,
                "n_mels": mel_cfg.n_mels,
                "f_min": mel_cfg.f_min,
                "f_max": mel_cfg.f_max,
                "sample_rate": mel_cfg.sample_rate,
                "n_fft": mel_cfg.n_fft,
                "hop_length": mel_cfg.hop_length,
                "architecture": "causal_streaming",
            },
        }

        # Safety checkpoint (overwritten each epoch)
        torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")
            print(f"  * New best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CW-Former (Conformer CW decoder)")
    parser.add_argument("--scenario", type=str, default="clean",
                        choices=["test", "clean", "moderate", "full"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_cwformer")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp",
                        help="Disable AMP (mixed precision). Auto-disabled on ROCm.")
    parser.add_argument("--lr-resume", action="store_true", dest="lr_resume",
                        help="Resume LR schedule from checkpoint state. Without "
                             "this, a fresh cosine schedule spans the remaining "
                             "epochs.")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Micro-batch size (gradient accumulation to effective ~64)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="DataLoader workers (default: min(8, cpu_count))")
    parser.add_argument("--max-audio-sec", type=float, default=15.0,
                        help="Max audio duration per sample (seconds)")
    parser.add_argument("--reuse-factor", type=int, default=1, dest="reuse_factor",
                        help="Replay each generated data buffer this many times before "
                             "regenerating. 1=disabled (generate fresh each epoch). "
                             "Recommended: 10 for moderate/full.")
    parser.add_argument("--buffer-epochs", type=int, default=1, dest="buffer_epochs",
                        help="Number of generation passes per buffer fill. Each pass "
                             "produces epoch_size fresh samples, so buffer holds "
                             "buffer_epochs * epoch_size unique samples. "
                             "Recommended: 3 with --cache-dir (audio is ~4 GB/pass). "
                             "Set reuse-factor >= buffer-epochs for good speedup.")
    parser.add_argument("--cache-dir", type=str, default=None, dest="cache_dir",
                        help="Directory for disk-based buffer cache. Required when "
                             "buffer_epochs > 1 for audio (each pass ~3-4 GB). "
                             "Batches are written as .pt files and reused across "
                             "replay passes; cleaned up before each refill. "
                             "Example: --cache-dir /tmp/cwformer_cache")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
