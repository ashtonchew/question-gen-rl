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

## Exporting the Trained Model

After training, merge LoRA adapters with the base model and export as a standalone HuggingFace model:

```bash
# Install export dependencies
pip install -e ".[export]"

# Export from checkpoint
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --verify
```

## Using the Exported Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/qwen3-4b-question-gen")
tokenizer = AutoTokenizer.from_pretrained("exports/qwen3-4b-question-gen")

inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

Or with vLLM for faster inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="exports/qwen3-4b-question-gen")
outputs = llm.generate(["Your prompt here"], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
```

## Uploading to HuggingFace Hub

Push your trained model to HuggingFace Hub:

```bash
# Set your HF credentials in .env
# HF_TOKEN=hf_xxxxx
# HF_REPO=your-username/your-model-name

# Export and upload
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub
```

Or specify the repo directly:

```bash
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --hub_repo your-username/your-model-name
```

## Testing the Inference Endpoint

After deploying to HuggingFace Inference Endpoints:

```bash
# Set your endpoint URL in .env
# HF_ENDPOINT_URL=https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud

# Test the endpoint
python scripts/test_endpoint.py

# Or with a custom prompt
python scripts/test_endpoint.py --prompt "Generate a screening question for a DevOps engineer:"
```

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
│   ├── format_prompts.py      # raw parquet → formatted parquet
│   ├── export_merged_model.py # Export trained model to HuggingFace format
│   └── test_endpoint.py       # Test HuggingFace inference endpoint
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
| `XAI_API_KEY` | API key for Grok judge (required for training) |
| `HF_TOKEN` | HuggingFace API token (required for --push_to_hub and endpoint) |
| `HF_REPO` | HuggingFace repository name, e.g., `username/model-name` |
| `HF_ENDPOINT_URL` | HuggingFace Inference Endpoint URL |
