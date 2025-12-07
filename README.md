# Question Generation RL

Train a language model to generate technical screening questions using reinforcement learning with LLM-as-judge or human feedback rewards.

**What it does:**
- Takes a role description (title, skills, tech stack) → generates a targeted screening question
- Trains Qwen3-4B using GRPO (Group Relative Policy Optimization)
- Supports both automated (LLM judge) and human-in-the-loop feedback

```bash
# Quick start
python scripts/prepare_dataset.py      # 1. Create data (once)
python -m src.recruiter.main           # 2. Train
python scripts/eval.py --model rl      # 3. Evaluate
```

---

## Setup

**Recommended:** Spin up a GPU instance on [Lambda Labs](https://lambdalabs.com/), [RunPod](https://runpod.io/), or your cloud provider of choice. Training requires a GPU with 24GB+ VRAM (A10, A100, or similar).

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)
```

---

## Training Modes

### Offline Mode (LLM Judge) — Default

Uses Grok to automatically score generated questions. Fast, no human involvement.

```bash
python -m src.recruiter.main --config-name=offline_config
```

| Setting | Value |
|---------|-------|
| Config | `configs/offline_config.yaml` |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Reward source | Grok LLM judge |
| Batch size | 64 |

### Online Mode (Human Feedback)

Uses human ratings from the feedback UI. Start the UI first, then train.

```bash
# Terminal 1: Start feedback UI
cd feedback-ui && uvicorn app:app --port 8000

# Terminal 2: Start training
python -m src.recruiter.main --config-name=online_config
```

| Setting | Value |
|---------|-------|
| Config | `configs/online_config.yaml` |
| Base model | `exports/qwen3-4b-question-gen` (pre-trained) |
| Reward source | Human feedback API |
| Batch size | 16 (smaller, humans are the bottleneck) |

---

## Human Feedback UI

Web interface for rating AI-generated questions. Used for online RL training.

```bash
cd feedback-ui
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000 to:
- Rate questions on relevance, clarity, and discriminative power (1-5 scale)
- View statistics and recent ratings at `/stats`
- Export feedback data via `/api/export?format=json` or `?format=parquet`

**Queue API** (for online training integration):
- `POST /api/queue/submit` — Submit question for rating
- `GET /api/queue/result/{id}` — Poll for rating result
- `GET /api/queue/pending` — List pending questions

---

## Evaluation

```bash
# Single model
python scripts/eval.py --model rl

# Compare multiple models
python scripts/eval.py --model baseline rl grok-4-1

# Use local checkpoint
python scripts/eval.py --model rl --checkpoint exports/my-model
```

**Available models:**

| Alias | Model | Provider |
|-------|-------|----------|
| `baseline` | Qwen3-4B-Instruct | Local (vLLM) |
| `rl` | ash256/qwen3-4b-question-gen | Local (auto-downloads) |
| `grok-4-1` | grok-4-1-fast-non-reasoning | xAI API |
| `claude-4-5-haiku` | claude-haiku-4-5-20251001 | Anthropic API |
| `gpt-5-mini` | gpt-5-mini-2025-08-07 | OpenAI API |

**Metrics** (scored 0-10 by LLM judge):
- **Relevance** — Does the question test skills needed for the role?
- **Clarity** — Is the question unambiguous and well-formed?
- **Discriminative** — Does it distinguish good candidates from weak ones?
- **Composite** — Average of the three

---

## Model Export & Deployment

### Export from Checkpoint

Merge LoRA adapters with base model:

```bash
pip install -e ".[export]"

python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --verify
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max_model_len` | 8192 | Max context length (vLLM compatible) |
| `--torch_dtype` | bfloat16 | Model dtype |
| `--verify` | — | Run generation test after export |
| `--push_to_hub` | — | Upload to HuggingFace Hub |
| `--create_endpoint` | — | Create inference endpoint |

### Download Pre-trained Model

```bash
python scripts/download_model.py
# Downloads to: exports/qwen3-4b-question-gen
```

### Push to HuggingFace Hub

```bash
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --hub_repo your-username/your-model-name
```

### Deploy to HuggingFace Inference Endpoints

```bash
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_100 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --create_endpoint
```

Creates an endpoint with:
- vLLM v0.8.5 (T4/L4 compatible)
- Nvidia T4 GPU ($0.50/hr)
- Scale-to-zero after 1 hour

Use `--instance_type nvidia-l4` for larger models.

### Test Inference Endpoint

```bash
python scripts/test_endpoint.py
python scripts/test_endpoint.py --prompt "Generate a screening question for a DevOps engineer:"
```

### Using the Exported Model

**With Transformers:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/qwen3-4b-question-gen")
tokenizer = AutoTokenizer.from_pretrained("exports/qwen3-4b-question-gen")

inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

**With vLLM (faster):**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="exports/qwen3-4b-question-gen")
outputs = llm.generate(["Your prompt here"], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
```

---

## Data Pipeline

```
data/backend_roles.json  →  data/raw/*.parquet  →  data/processed/*.parquet  →  training
     (500 roles)              (role data)            (formatted prompts)
                                   ↑                        ↑
                          prepare_dataset.py          auto-formatted
                             (run once)              on training start
```

**Prompt formatting** lives in `src/recruiter/prompts.py`. Edit it and training auto-detects changes.

---

## Project Structure

```
├── configs/
│   ├── train_config.yaml         # Base training config
│   ├── offline_config.yaml       # LLM judge mode
│   └── online_config.yaml        # Human feedback mode
├── data/
│   ├── backend_roles.json        # 500 role definitions
│   ├── raw/                      # Raw parquet (role data)
│   └── processed/                # Formatted parquet (with prompts)
├── feedback-ui/                  # Human feedback web UI
│   ├── app.py                    # FastAPI server
│   ├── database.py               # SQLAlchemy models
│   ├── generator.py              # Question generation
│   └── templates/                # Jinja2 templates
├── scripts/
│   ├── prepare_dataset.py        # JSON → raw parquet
│   ├── format_prompts.py         # Raw → formatted parquet
│   ├── eval.py                   # Model evaluation
│   ├── export_merged_model.py    # Export to HuggingFace format
│   ├── download_model.py         # Download from Hub
│   ├── test_endpoint.py          # Test inference endpoint
│   └── analyze_results.py        # Results visualization
├── src/recruiter/
│   ├── main.py                   # Training entrypoint
│   ├── env.py                    # SkyRL environments
│   ├── reward.py                 # Grok judge reward
│   ├── prompts.py                # Prompt formatting
│   ├── human_feedback.py         # Human feedback client
│   └── metrics_logger.py         # JSON metrics logging
└── results/
    └── eval_results.json         # Evaluation outputs
```

---

## Configuration

Key settings in `configs/train_config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `trainer.policy.model.path` | `exports/qwen3-4b-question-gen` | Base model path |
| `trainer.policy.lora.rank` | 8 | LoRA rank |
| `trainer.policy.optimizer_config.lr` | 1e-5 | Learning rate |
| `trainer.train_batch_size` | 64 | Batch size |
| `trainer.epochs` | 5 | Training epochs |
| `generator.n_samples_per_prompt` | 4 | Samples per role for GRPO |
| `generator.sampling_params.max_generate_length` | 256 | Max tokens to generate |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | Yes | Grok API key (training judge + evaluation) |
| `ANTHROPIC_API_KEY` | For Claude eval | Anthropic API key |
| `OPENAI_API_KEY` | For GPT eval | OpenAI API key |
| `HF_TOKEN` | For Hub upload | HuggingFace API token |
| `HF_REPO` | For Hub upload | HuggingFace repo name (e.g., `username/model`) |
| `HF_ENDPOINT_URL` | For endpoint test | Inference endpoint URL |
