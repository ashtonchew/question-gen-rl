#!/bin/bash
# Training launcher script for question generation RL
set -e

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XAI_API_KEY=${XAI_API_KEY:?"XAI_API_KEY must be set"}

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Question Generation RL Training ==="
echo "Project directory: $PROJECT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Prepare dataset if not exists
if [ ! -f "data/processed/train.parquet" ]; then
    echo "Preparing dataset..."
    python scripts/prepare_dataset.py
    echo ""
fi

# Run training
echo "Starting training..."
python -m src.recruiter.main "$@"
