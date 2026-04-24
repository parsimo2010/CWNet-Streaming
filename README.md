# CWformer

A causal streaming neural Morse code (CW) decoder. It uses a fully causal Conformer architecture (~19.5M parameters) with CTC loss that processes audio left-to-right with no bidirectional attention, eliminating the window-stitching artifacts of the original [CWNet](https://github.com/parsimo2010/CWNet) bidirectional model.

CWformer decodes CW from audio in real time — feed it audio from a USB sound card, a file, or stdin, and it emits decoded text as characters are confirmed. It targets 15–40 WPM across all common key types (straight key, bug, paddle, cootie) at SNR > 5–8 dB, with under 2.5 seconds of latency from audio to character emission. Current results show it performs decently down to 10 WPM and it has been tested up to 35 WPM with good accuracy, suggesting it can meet the 40 WPM goal. Accuracy degrades progressively worse as speed drops below 10 WPM, and at some high speeed the accuracy will degrade due to the 20ms time resolution of the inputs.

## Who should do what

- **If you just want to decode CW:** download the latest release (a zipped ONNX model + a small inference script) and follow "Running the decoder" below. You do not need this repository.
- **If you want to train your own model or hack on the architecture:** clone this repo and follow "Training your own model".

## How It Works

Audio is processed causally: each frame only sees past context, never the future. During inference, model state (KV caches and convolution buffers) carries forward between processing chunks, so there are no windows to stitch together. The model is trained so that characters sharing element prefixes (E=`.`, I=`..`, S=`...`, H=`....`, 5=`.....`) are held until the inter-character space is confirmed, ensuring that only correct decodes are emitted.

```
Audio (16 kHz mono)
  → Incremental log-mel spectrogram (40 bins, 25ms/10ms)
  → Causal ConvSubsampling (2× time reduction → 50 fps)
  → 12× Causal Conformer blocks (d=256, 4 heads, conv kernel=63)
  → CTC head → greedy decode
  → Text
```

## Running the decoder (ONNX release)

This path uses the pre-trained ONNX model. It runs on a Raspberry Pi 5, a laptop, or anything with Python 3.10+.

### 1. Download and unpack the latest release

```bash
mkdir -p ~/cwformer && cd ~/cwformer
curl -L -o cwformer-onnx.zip https://github.com/parsimo2010/CWformer/releases/latest/download/cwformer-onnx.zip
unzip -o cwformer-onnx.zip
```

The zip always lives at the same URL, so this one-liner never changes between releases.

### 2. Run the install script

```bash
./install.sh
```

On Raspberry Pi OS (Bookworm) or Debian/Ubuntu this installs the required apt packages (`python3-venv`, `python3-dev`, `libsndfile1`, `libportaudio2`), creates a self-contained Python venv at `./venv`, and installs the runtime pip deps from `requirements-deploy.txt` (`numpy`, `soundfile`, `onnxruntime`, `sounddevice`, `scipy`). No PyTorch needed.

### 3. Decode

Activate the venv, then pick a mode:

```bash
source venv/bin/activate
```

**Streaming from a USB sound card.** Find the device index, then start decoding:

```bash
python inference_onnx.py --model cwformer_streaming_int8.onnx --list-devices
python inference_onnx.py --model cwformer_streaming_int8.onnx --device 2
```

Omit the index to use the system default input:

```bash
python inference_onnx.py --model cwformer_streaming_int8.onnx --device
```

**Streaming from stdin.** Useful for chaining with `arecord`, `sox`, `rtl_fm`, or any tool that produces raw PCM:

```bash
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -t raw | \
  python inference_onnx.py --model cwformer_streaming_int8.onnx --stdin
```

**Decoding a WAV file.**

```bash
python inference_onnx.py --model cwformer_streaming_int8.onnx --input recording.wav
```

**Using the FP32 model** (slightly more accurate, slower on CPU):

```bash
python inference_onnx.py --model cwformer_streaming_fp32.onnx --input recording.wav
```

## Training your own model

You need a machine with a CUDA-capable GPU and the cloned repo. AMD GPUs work via ROCm but are not speed-optimized or fully tested.

### 1. Clone and set up the venv

```bash
git clone https://github.com/parsimo2010/CWformer.git
cd CWformer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
```

`requirements-train.txt` lists `torch`, `torchaudio`, `numpy`, `scipy`, `soundfile`, `tqdm`, `onnx`, `onnxruntime`. If you need a particular PyTorch variant (CUDA version, ROCm, CPU-only), install that first with the command from [pytorch.org](https://pytorch.org/get-started/locally/) and then run the `-r requirements-train.txt` step — pip will keep your existing torch install.

### 2. Train through the curriculum

Training uses synthetic Morse audio generated on the fly — no dataset download is needed. Training follows a three-stage curriculum. Each stage adds progressively harder conditions (lower SNR, wider WPM range, more augmentations). You may want to stop early on the clean and moderate stages once validation loss plateaus — additional epochs past that point won't help.

**Stage 1 — Clean** (high SNR, moderate speeds):

```bash
python -m neural_decoder.train_cwformer --scenario clean --ckpt-dir checkpoints_clean
```

**Stage 2 — Moderate** (medium SNR, timing variance, QRM):

```bash
python -m neural_decoder.train_cwformer --scenario moderate \
  --checkpoint checkpoints_clean/best_model.pt \
  --ckpt-dir checkpoints_moderate
```

**Stage 3 — Full** (low SNR, all augmentations, all key types):

```bash
python -m neural_decoder.train_cwformer --scenario full \
  --checkpoint checkpoints_moderate/best_model.pt \
  --ckpt-dir checkpoints_full
```

You can also let the training loop advance through the stages automatically on CER plateau:

```bash
python -m neural_decoder.train_cwformer --scenario clean --auto-curriculum
```

### 3. Export to ONNX for deployment

```bash
python quantize_cwformer.py --checkpoint checkpoints_full/best_model.pt --output-dir deploy/
```

This produces `cwformer_streaming_fp32.onnx`, `cwformer_streaming_int8.onnx`, `mel_config.json`, `mel_basis.npy`, and `mel_window.npy` in the `deploy/` directory.

### 4. Benchmark

```bash
python benchmark_cwformer.py --checkpoint checkpoints_full/best_model.pt --csv results.csv
```

Or run the random-parameter sweep across the full augmentation distribution:

```bash
python benchmark_random_sweep.py --checkpoint checkpoints_full/best_model.pt --n 5000
```

## Project Structure

```
CWformer/
├── config.py                    # MorseConfig, TrainingConfig
├── vocab.py                     # CTC vocabulary (52 classes)
├── metrics.py                   # Shared Levenshtein / CER helpers
├── morse_table.py               # ITU Morse code table + binary trie
├── morse_generator.py           # Synthetic training data generation
├── qso_corpus.py                # Realistic ham radio QSO text corpus
├── quantize_cwformer.py         # ONNX export + INT8 quantization
├── benchmark_cwformer.py        # Structured SNR×WPM×key benchmark
├── benchmark_random_sweep.py    # Random parameter sweep benchmark
├── requirements-deploy.txt      # Runtime deps for ONNX inference
├── requirements-train.txt       # Training deps (torch + torchaudio)
├── install.sh                   # Raspberry Pi / Debian install script (bundled in the ONNX release zip)
│
├── neural_decoder/
│   ├── cwformer.py              # CW-Former model (forward + forward_streaming)
│   ├── conformer.py             # Causal Conformer blocks (attention, conv, state)
│   ├── mel_frontend.py          # Mel spectrogram (batch + streaming)
│   ├── rope.py                  # Rotary Position Embeddings
│   ├── dataset_audio.py         # Streaming IterableDataset
│   ├── train_cwformer.py        # Training loop (curriculum learning)
│   └── inference_cwformer.py    # CWFormerStreamingDecoder
│
├── deploy/
│   └── inference_onnx.py        # ONNX Runtime streaming inference
│
├── tests/
│   ├── test_streaming_equivalence.py   # forward vs forward_streaming diff
│   └── diagnostic/                      # Ad-hoc PyTorch-vs-ONNX diff tools
│
└── recordings/                  # Real HF band noise recordings for augmentation
```

## Future work

Open items I'd like to get to (help welcome):

- **CI.** GitHub Actions that at minimum runs `tests/test_streaming_equivalence.py` on a tiny synthetic checkpoint and verifies `tests/diagnostic/` scripts still import. A full benchmark in CI would be great eventually but needs a trained model fixture.

## Authorship

The code in this repository was primarily written by [Claude Code](https://claude.ai/claude-code) (Anthropic), with architecture design, domain expertise, and direction from [Harris Butler](https://github.com/parsimo2010) (parsimo2010).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
