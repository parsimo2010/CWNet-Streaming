#!/usr/bin/env bash
# CWformer deploy install script. Run from the directory this script lives in,
# after unzipping cwformer-onnx.zip. Sets up a self-contained Python venv and
# installs the runtime dependencies for ONNX inference.
#
# Target: Raspberry Pi OS (Bookworm), Debian, or Ubuntu with internet access.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing system packages (sudo apt)"
sudo apt update
sudo apt install -y python3-venv python3-dev libsndfile1 libportaudio2

echo "==> Creating Python venv at ./venv"
if [ ! -d venv ]; then
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements-deploy.txt

cat <<'EOF'

==> Done.

To decode CW, activate the venv and run inference_onnx.py. Examples:

    source venv/bin/activate

    # List audio input devices
    python inference_onnx.py --model cwformer_streaming_int8.onnx --list-devices

    # Stream from a specific device (e.g. a USB sound card at index 2)
    python inference_onnx.py --model cwformer_streaming_int8.onnx --device 2

    # Stream from the default device
    python inference_onnx.py --model cwformer_streaming_int8.onnx --device

    # Decode a WAV file
    python inference_onnx.py --model cwformer_streaming_int8.onnx --input recording.wav

Swap cwformer_streaming_int8.onnx for cwformer_streaming_fp32.onnx for the
full-precision model (slightly more accurate, slower on CPU).
EOF
