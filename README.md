# Question Generation RL

Train a language model to generate technical screening questions using reinforcement learning with LLM-as-judge rewards.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your XAI_API_KEY for the Grok judge
```

## Quick Start

```bash
# 1. Create raw data (once, or when adding new roles)
python scripts/prepare_dataset.py

# 2. Train (auto-formats prompts if needed)
python -m src.recruiter.main
```

That's it. Prompt formatting happens automatically when training starts.

## Changing Prompts

All prompt logic lives in `src/recruiter/prompts.py`. To change the prompt format:

1. Edit `src/recruiter/prompts.py`
2. Run `python -m src.recruiter.main`

The training script auto-detects when `prompts.py` changes and reformats data before training.

## Data Pipeline

```
data/backend_roles.json  →  data/raw/*.parquet  →  data/processed/*.parquet  →  training
                              (raw role data)       (formatted prompts)
                                    ↑                       ↑
                         prepare_dataset.py          auto-formatted
                            (run once)              on training start
```

## Project Structure

```
├── configs/
│   └── train_config.yaml      # Training hyperparameters
├── data/
│   ├── backend_roles.json     # Source role definitions
│   ├── raw/                   # Raw parquet (role data only)
│   └── processed/             # Formatted parquet (with prompts)
├── scripts/
│   ├── prepare_dataset.py     # JSON → raw parquet
│   └── format_prompts.py      # raw parquet → formatted parquet
├── src/recruiter/
│   ├── main.py                # Training entrypoint
│   ├── env.py                 # SkyRL environment
│   ├── reward.py              # Grok judge reward function
│   └── prompts.py             # Prompt formatting (edit this!)
└── Makefile
```

## Configuration

Key settings in `configs/train_config.yaml`:

- `trainer.policy.model.path`: Base model (default: Qwen/Qwen3-4B-Instruct-2507)
- `generator.n_samples_per_prompt`: Samples per prompt for GRPO (default: 4)
- `generator.sampling_params.max_generate_length`: Max tokens to generate
- `trainer.train_batch_size`: Batch size for training

## Environment Variables

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | API key for Grok judge (required) |
