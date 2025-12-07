#!/bin/bash
set -e

echo "=== Question Gen RL Setup ==="

# Get script directory BEFORE changing dirs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone SkyRL (needed as dependency)
cd ~
if [ ! -d "SkyRL" ]; then
    git clone https://github.com/NovaSky-AI/SkyRL.git
fi

# Setup your project
cd "$SCRIPT_DIR"
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install torch first (required for flash-attn build)
uv pip install torch==2.8.0

# Install SkyRL from local clone (--no-build-isolation for flash-attn)
uv pip install -e ~/SkyRL/skyrl-train[vllm] --no-build-isolation
uv pip install -e ~/SkyRL/skyrl-gym

# Install project (ensure we're in project directory)
cd "$SCRIPT_DIR"
uv pip install -e .

# Prepare dataset
python scripts/prepare_dataset.py --input data/backend_roles.json --output_dir data/processed

# Reminder for API key
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Before running, set your Grok API key:"
echo "  export XAI_API_KEY='your-key-here'"
echo ""
echo "Then run training:"
echo "  python -m src.recruiter.main"
