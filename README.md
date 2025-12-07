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

## Downloading a Trained Model

Download the RL-trained model from HuggingFace Hub for evaluation or continued training:

```bash
# Download model to exports/ directory
python scripts/download_model.py
# Downloads to: exports/qwen3-4b-question-gen
```

The script uses the `HF_REPO` environment variable (default: `ash256/qwen3-4b-question-gen`).

### For Online RL (Continued Training)

After downloading, update your training config to start from the trained model:

```yaml
# configs/train_config.yaml
trainer:
  policy:
    model:
      path: "exports/qwen3-4b-question-gen"  # Use downloaded model as base
```

Then run training as usual:

```bash
python -m src.recruiter.main
```

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

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max_model_len` | 8192 | Max context length (vLLM compatible). Set to 0 to keep original. |
| `--torch_dtype` | bfloat16 | Model dtype: float32, float16, or bfloat16 |
| `--verify` | - | Run generation test after export |
| `--push_to_hub` | - | Upload to HuggingFace Hub |
| `--create_endpoint` | - | Create inference endpoint after push |

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

## Deploying to HuggingFace Inference Endpoints

Create an inference endpoint programmatically after pushing to Hub:

```bash
# Push to Hub AND create endpoint (T4 GPU, vLLM v0.8.5)
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --hub_repo your-username/your-model-name \
    --create_endpoint
```

This creates an endpoint with:
- **vLLM v0.8.5** (stable for T4/older GPUs)
- **Nvidia T4** GPU (16GB, $0.50/hr)
- **Public access** (no authentication required)
- **Scale-to-zero** after 1 hour of inactivity

### Endpoint Options

```bash
# Use L4 GPU instead of T4 (better for larger models)
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --create_endpoint \
    --instance_type nvidia-l4

# Custom endpoint name
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --create_endpoint \
    --endpoint_name my-question-gen
```

### Manual Endpoint Creation

If creating manually via the HuggingFace UI, use these settings for T4 compatibility:
- **vLLM version**: `vllm/vllm-openai:v0.8.5` (v0.11.0 has bugs on T4)
- **Instance**: Nvidia T4 or L4
- **max_position_embeddings** in config.json should be ≤8192 (set automatically by export script)

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

## Evaluation

Evaluate and compare models on the held-out test set:

```bash
# Set API keys
export XAI_API_KEY=your-xai-key        # Required (for judge + Grok eval)
export ANTHROPIC_API_KEY=your-key      # For Claude eval
export OPENAI_API_KEY=your-key         # For GPT eval

# Generate test set (if not already done)
python scripts/prepare_dataset.py
python scripts/format_prompts.py

# Evaluate single model
python scripts/eval.py --model grok-4-1

# Evaluate multiple models (parallelized)
python scripts/eval.py --model grok-4-1 claude-4-5-haiku gpt-5-mini

# Evaluate RL model (auto-downloads from HuggingFace Hub)
python scripts/eval.py --model rl

# Compare baseline vs RL
python scripts/eval.py --model baseline rl

# Use a local checkpoint instead (optional)
python scripts/eval.py --model rl --checkpoint exports/my-local-model
```

**Available models:**
| Alias | Model | Provider |
|-------|-------|----------|
| `grok-4-1` | grok-4-1-fast-non-reasoning | xAI |
| `claude-4-5-haiku` | claude-haiku-4-5-20251001 | Anthropic |
| `gpt-5-mini` | gpt-5-mini-2025-08-07 | OpenAI |
| `baseline` | Qwen3-4B-Instruct | Local (vLLM) |
| `rl` | ash256/qwen3-4b-question-gen | Local (vLLM, auto-downloads) |

**Metrics** (scored 0-10 by LLM judge):
- **Relevance**: Does the question test skills needed for the role?
- **Clarity**: Is the question unambiguous and well-formed?
- **Discriminative Power**: Does it distinguish good candidates from weak ones?
- **Composite**: Average of the three metrics

The judge model is always `grok-4-1-fast-non-reasoning` for fair comparison.

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
│   ├── download_model.py      # Download model from HuggingFace Hub
│   ├── eval.py                # Model evaluation on test set
│   ├── analyze_results.py     # Results analysis and visualization
│   └── test_endpoint.py       # Test HuggingFace inference endpoint
├── src/recruiter/
│   ├── main.py                # Training entrypoint
│   ├── env.py                 # SkyRL environment
│   ├── reward.py              # Grok judge reward function
│   ├── prompts.py             # Prompt formatting (edit this!)
│   ├── dataset.py             # Dataset loading
│   └── schemas.py             # Pydantic schemas
├── results/
│   └── eval_results.json      # Evaluation results
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
| `XAI_API_KEY` | API key for Grok (required for training + evaluation judge) |
| `ANTHROPIC_API_KEY` | API key for Claude evaluation |
| `OPENAI_API_KEY` | API key for GPT evaluation |
| `HF_TOKEN` | HuggingFace API token (required for --push_to_hub and endpoint) |
| `HF_REPO` | HuggingFace repository name, e.g., `username/model-name` |
| `HF_ENDPOINT_URL` | HuggingFace Inference Endpoint URL |
