# Online RL Add-on Guide (MVP)

**Version:** 2.0
**Last Updated:** 2025-12-07
**Prerequisite:** Existing question generation system
**Time to integrate:** 2-4 hours

---

## 1. Overview

This guide shows you how to add **online reinforcement learning** to your existing LLM-based generation system. Instead of building from scratch, you'll add three components that enable GRPO (Group Relative Policy Optimization) training.

### What You'll Add

```
Your Existing System          +  Online RL Components
─────────────────────         ─────────────────────────
[Dataset]                     [reward.py]  - LLM-as-judge
[LLM Generation]       →      [env.py]     - SkyRL environment
[Output]                      [config]     - Training parameters
```

### The Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│  For each prompt in dataset:                                │
│    1. Model generates output                                │
│    2. Judge LLM scores the output (0-10 on your criteria)   │
│    3. Reward signal updates model weights                   │
│    4. Model improves at your specific task                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites Checklist

Before starting, ensure you have:

### Infrastructure
- [ ] **GPU**: Single GPU with 24GB+ VRAM (A10, A100, or RTX 4090)
- [ ] **Python**: 3.12+
- [ ] **CUDA**: 12.1+

### Your Existing System
- [ ] **Working LLM**: Can generate outputs (local vLLM, HuggingFace, or API)
- [ ] **Dataset**: Prompts you want to train on
- [ ] **Judge API**: Access to Grok, Claude, or GPT-4 for scoring

### Dependencies
```bash
# Install SkyRL
pip install skyrl-train[vllm] skyrl-gym

# Or from source
git clone https://github.com/NovaSky-AI/SkyRL.git
pip install -e SkyRL/skyrl-train[vllm]
pip install -e SkyRL/skyrl-gym
```

---

## 3. Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dataset    │────▶│    Model     │────▶│   Output     │
│  (parquet)   │     │  (Qwen, etc) │     │ (generated)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌───────────────────────────┘
                     ▼
              ┌──────────────┐     ┌──────────────┐
              │    Judge     │────▶│   Reward     │
              │  (Grok/GPT)  │     │   (0-1)      │
              └──────────────┘     └──────┬───────┘
                                          │
                     ┌────────────────────┘
                     ▼
              ┌──────────────┐
              │    GRPO      │───▶ Model weights updated
              │   Update     │
              └──────────────┘
```

### Components You'll Create

| File | Purpose | Lines |
|------|---------|-------|
| `reward.py` | LLM-as-judge with JSON sanitization | 171 |
| `env.py` | SkyRL environment wrapper | 119 |
| `main.py` | Training entrypoint with auto-formatting | 104 |
| `prompts.py` | Prompt formatting for roles | 51 |
| `config.yaml` | Training parameters | ~70 |

---

## 4. Core Components

### 4.1 Reward Function (`src/recruiter/reward.py`)

The repo ships a Grok-based judge with robust JSON parsing. The implementation includes:

1. **68-line scoring system prompt** with detailed rubrics for relevance, clarity, and discriminative power (0-10 each)
2. **JSON sanitization utilities** to handle markdown wrappers and control characters
3. **Pydantic validation** via `JudgeResponse` schema

#### Key Functions

```python
def sanitize_json_string(text: str) -> str:
    """Remove control characters (0x00-0x1F) that break JSON parsing.
    Replaces tabs/newlines with spaces, skips null chars."""

def extract_json(text: str) -> str:
    """Extract JSON object from text, handling:
    - Markdown code blocks (```json ... ```)
    - Trailing content after JSON
    - Uses json.JSONDecoder.raw_decode() for robust parsing"""

def judge_question(role_description: str, question: str) -> Tuple[float, dict]:
    """Call Grok API to score a question. Returns (reward, details_dict)."""
```

#### Scoring Rubric (from JUDGE_SYSTEM_PROMPT)

| Criterion | Score Range | Description |
|-----------|-------------|-------------|
| **Relevance** | 0-10 | Does the question test skills needed for the role? |
| **Clarity** | 0-10 | Is the question unambiguous and well-formed? |
| **Discriminative** | 0-10 | Would this distinguish good from weak candidates? |

The prompt includes red flags that subtract points (e.g., generic questions -3, bundled questions -2, yes/no questions -2).

#### Main Function

```python
def judge_question(role_description: str, question: str) -> Tuple[float, dict]:
    if client is None:
        return 0.0, {"error": "XAI_API_KEY not set"}

    chat = client.chat.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[system(JUDGE_SYSTEM_PROMPT)],
        response_format="json_object"  # Force JSON output
    )
    chat.append(user(user_prompt))

    response = chat.sample()
    json_str = extract_json(response.content)  # Sanitize and extract
    scores = JudgeResponse.model_validate_json(json_str)
    reward = (scores.relevance + scores.clarity + scores.discriminative) / 30.0
    return reward, scores.model_dump()
```

#### Configuration

- **Env var**: `XAI_API_KEY` (required)
- **Model**: `grok-4-1-fast-non-reasoning` (hardcoded)
- **Timeout**: 600s via `xai-sdk` (built-in retries)
- **Reward range**: 0.0 to 1.0 (normalized from 0-30 total score)

#### Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `reward=0.0` with `{"error": "XAI_API_KEY not set"}` | Missing API key | Set `XAI_API_KEY` |
| `reward=0.0` with exception details | API error | Check logs, verify API access |
| Constant ~0.5 reward | No learning signal | Verify judge prompt is working |

### 4.2 Environment Wrapper (`src/recruiter/env.py`)

Matches the shipped SkyRL adapter and uses the judge above.

```python
"""SkyRL environment for technical question generation."""
import re
from dataclasses import dataclass
from typing import Optional
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from .reward import judge_question

@dataclass
class QuestionGenEnvConfig:
    max_question_length: int = 500
    min_question_length: int = 20

class QuestionGenEnv(BaseTextEnv):
    def __init__(self, config: Optional[QuestionGenEnvConfig] = None):
        super().__init__()
        self.config = config or QuestionGenEnvConfig()
        self._prompt = None
        self._role_id = None

    def _extract_prompt_text(self, prompt) -> str:
        # Dataset provides a chat list; grab the user message content.
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return str(prompt)

    def _extract_role_id(self, prompt_text: str) -> Optional[str]:
        match = re.search(r'\*\*ID:\*\*\s*(\S+)', prompt_text)
        return match.group(1) if match else None

    def init(self, prompt):
        self._prompt = self._extract_prompt_text(prompt)
        self._role_id = self._extract_role_id(self._prompt)
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        question = action.strip()
        if len(question) < self.config.min_question_length:
            return BaseTextEnvStepOutput(observations=[], reward=-0.5, done=True, metadata={"error": "Question too short"})
        if len(question) > self.config.max_question_length:
            question = question[:self.config.max_question_length]
        if not self._prompt:
            return BaseTextEnvStepOutput(observations=[], reward=-1.0, done=True, metadata={"error": "No prompt available"})

        reward, judge_details = judge_question(role_description=self._prompt, question=question)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"question": question, "judge_scores": judge_details, "role_id": self._role_id},
        )
```

Note: `BaseTextEnvStepOutput` uses `done`/`observations` (not `terminated`/`truncated`).

Runtime behavior worth calling out:
- Single turn only (`done=True` after one `step`).
- Validation: questions <20 chars → `reward=-0.5`; missing prompt → `reward=-1.0`; >500 chars are truncated before judging.
- Metadata includes `question`, `judge_scores`, and `role_id` for downstream logging.

### 4.3 Training Entrypoint (`src/recruiter/main.py`)

The entrypoint includes **automatic data formatting** that regenerates processed data when needed.

#### Auto-Formatting Logic

```python
def ensure_data_formatted():
    """Auto-runs before training. Regenerates processed data if:
    - data/processed/*.parquet doesn't exist, OR
    - data/raw/*.parquet is newer than processed, OR
    - src/recruiter/prompts.py is newer than processed
    """
```

This means you can modify `prompts.py` and re-run training—the formatted data will auto-update.

#### Environment Registration

```python
class QuestionGenEnvWrapper(QuestionGenEnv):
    """Wrapper for SkyRL registration. SkyRL passes prompt to init(), not __init__."""

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    register(
        id="question-gen",  # Must match environment.env_class in config
        entry_point="src.recruiter.main:QuestionGenEnvWrapper",
    )
    exp = BasePPOExp(cfg)
    exp.run()
```

#### Main Entry

```python
@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    ensure_data_formatted()  # Auto-format before Ray starts
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))
```

#### Run Training

```bash
python -m src.recruiter.main
```

#### Important Notes

- **Auto-formatting**: No need to manually run `format_prompts.py`—it happens automatically
- **Raw data required**: Must have `data/raw/*.parquet` (create with `scripts/prepare_dataset.py`)
- **Prompt format**: Must include `**ID:** <role_id>` token—`QuestionGenEnv` extracts this for metadata
- **Environment ID**: The `"question-gen"` ID must match `environment.env_class` in config

### 4.4 Alternative: Runtime Formatting (`src/recruiter/dataset.py`)

If you prefer to format prompts at load time instead of pre-generating:

```python
from src.recruiter.dataset import load_and_format_dataset

# Load raw parquets and format prompts on-the-fly
dataset = load_and_format_dataset(["data/raw/train.parquet"])
# Returns HuggingFace Dataset with 'prompt' column formatted from 'role_json'
```

This utility is useful for:
- Rapid iteration on prompt format without regenerating files
- Testing prompt changes before committing to processed data
- Alternative workflows that don't use `ensure_data_formatted()`

Note: The main training flow uses `ensure_data_formatted()` in `main.py` instead of this module.

---

## 5. Essential Configuration (mirrors `configs/train_config.yaml`)

Key defaults already present in the repo (synced to the current YAML):

```yaml
data:
  train_data: ["data/processed/train.parquet"]
  val_data: ["data/processed/test.parquet"]

trainer:
  placement:
    colocate_all: true
    colocate_policy_ref: true
    policy_num_nodes: 1
    policy_num_gpus_per_node: 1
    critic_num_nodes: 1
    critic_num_gpus_per_node: 1
    ref_num_nodes: 1
    ref_num_gpus_per_node: 1

  strategy: fsdp2
  sequence_parallel_backend: "ulysses"

  policy:
    model:
      path: "Qwen/Qwen3-4B-Instruct-2507"
      lora:
        rank: 8
        alpha: 16
        dropout: 0
        lora_sync_path: "/tmp/skyrl_lora_sync"
        target_modules: "all-linear"
        exclude_modules: null
    optimizer_config:
      lr: 1.0e-5
  algorithm:
    advantage_estimator: "grpo"
    kl_loss_coef: 0.001
  epochs: 5
  train_batch_size: 64
  policy_mini_batch_size: 16
  critic_mini_batch_size: 16
  micro_train_batch_size_per_gpu: 2
  micro_forward_batch_size_per_gpu: 2
  eval_before_train: true
  eval_interval: 2
  ckpt_path: "checkpoints/"
  project_name: "question-gen-rl"
  run_name: "qwen3-4b-grpo"
  logger: "console"

  fully_async:
    num_parallel_generation_workers: 64

generator:
  model_name: ${trainer.policy.model.path}
  model_dtype: "bfloat16"
  backend: "vllm"
  num_inference_engines: 1
  n_samples_per_prompt: 4
  gpu_memory_utilization: 0.5
  vllm_v1_disable_multiproc: true
  sampling_params:
    max_generate_length: 256
    temperature: 1.0

environment:
  env_class: "question-gen"   # must match register() id
  skyrl_gym:
    max_env_workers: 8
```

Hardware note: fsdp2 + Qwen 4B + `n_samples_per_prompt=4` + async workers can exceed 24 GB. Downscale in this order: (1) set `generator.n_samples_per_prompt=2`; (2) cut `trainer.train_batch_size` to 32 and `micro_train_batch_size_per_gpu` to 1; (3) lower `environment.skyrl_gym.max_env_workers` and `fully_async.num_parallel_generation_workers` (e.g., 8→2). Increase back only if VRAM allows.

<IMPORTANT>
**Continuing Online RL (Multiple Rounds)**

If you want to do additional rounds of online RL on an already-trained model, you must use the **exported merged model**, not the base model:

```bash
# 1. Export your RL-trained model (merges LoRA into base weights)
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_31 \
    --output exports/qwen3-4b-round1

# 2. Update config to use exported model as the new base
# configs/train_config.yaml
trainer:
  policy:
    model:
      path: "exports/qwen3-4b-round1"  # NOT "Qwen/Qwen3-4B-Instruct-2507"

# 3. Run another round of online RL
python -m src.recruiter.main
```

Why this matters:
- The checkpoint only contains LoRA adapters, not the full model
- Starting from the base model again would lose all previous RL improvements
- Each round builds on the previous round's learned weights
- You can stack as many rounds as needed: `Base → Round 1 → Round 2 → ...`
</IMPORTANT>

---

## 6. Integration Steps

### Step 1: Prepare Your Data

```bash
# Required: Create raw parquet from JSON
python scripts/prepare_dataset.py           # writes data/raw/train.parquet & test.parquet
```

**That's it!** The training script (`main.py`) will automatically format prompts when needed.

#### Optional: Manual Formatting

If you want to inspect formatted data before training:

```bash
# Optional: Pre-generate formatted prompts (auto-runs anyway)
python scripts/format_prompts.py            # writes data/processed/*.parquet
```

#### How Auto-Formatting Works

`main.py` calls `ensure_data_formatted()` before training starts. It will regenerate `data/processed/*.parquet` if:
- Processed files don't exist
- Raw data is newer than processed data
- `prompts.py` has been modified

#### Dataset Schema

| Stage | File | Contents |
|-------|------|----------|
| Raw | `data/raw/*.parquet` | `role_json` (serialized role dict) |
| Processed | `data/processed/*.parquet` | `prompt` (chat list with `**ID:** <role_id>`) |

The prompt format is a two-message chat list: `[{"role": "system", ...}, {"role": "user", ...}]` with `**ID:** <role_id>` embedded for metadata extraction.

### Step 2: Set Environment Variable

```bash
export XAI_API_KEY="your-grok-api-key"
```

### Step 3: Run Training

```bash
python -m src.recruiter.main
```

---

## 7. Evaluation

Compare your RL-trained model against baselines and SOTA models.

### Supported Models

| Alias | Model | Provider |
|-------|-------|----------|
| `grok-4-1` | grok-4-1-fast-non-reasoning | xAI |
| `claude-4-5-haiku` | claude-haiku-4-5-20251001 | Anthropic |
| `gpt-5-mini` | gpt-5-mini-2025-08-07 | OpenAI |
| `baseline` | Qwen/Qwen3-4B-Instruct-2507 | Local vLLM |
| `rl` | (checkpoint path) | Local vLLM |

### Judge Configuration

- **Model**: `grok-4-1-fast-non-reasoning` (hardcoded for fair comparison)
- **Metrics**: relevance, clarity, discriminative (0-10 each)
- **Composite**: Average of all three metrics

### Usage

```bash
# Evaluate a single model
python scripts/eval.py --model grok-4-1

# Compare multiple API models (runs in parallel)
python scripts/eval.py --model grok-4-1 claude-4-5-haiku gpt-5-mini

# Compare baseline vs RL-trained
python scripts/eval.py --model baseline rl --checkpoint checkpoints/global_step_6

# Evaluate all models on 100 samples
python scripts/eval.py --model all --num_samples 100
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test_data` | `data/processed/test.parquet` | Test data path |
| `--model` | `all` | Model(s) to evaluate |
| `--checkpoint` | - | Required for `rl` model |
| `--num_samples` | all | Limit evaluation samples |
| `--max_workers` | 10 | Parallel API workers |
| `--output` | `results/eval_results.json` | Results file |

### Parallel Execution

- **API models** (grok, claude, gpt): Run in parallel with ThreadPoolExecutor
- **Local models** (baseline, rl): Run sequentially (vLLM constraint)
- **Results merge**: Only specified models are updated; existing results preserved

### Environment Variables

```bash
export XAI_API_KEY="..."       # Required for Grok models and judging
export ANTHROPIC_API_KEY="..." # Required for Claude models
export OPENAI_API_KEY="..."    # Required for GPT models
```

---

## 8. Export & Deployment

Export LoRA adapters merged with base model for deployment.

### Basic Export

```bash
# Merge LoRA and save locally
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_6 \
    --output exports/qwen3-4b-question-gen \
    --verify
```

### Push to HuggingFace Hub

```bash
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_6 \
    --output exports/qwen3-4b-question-gen \
    --push_to_hub \
    --hub_repo username/model-name
```

### Create Inference Endpoint

```bash
# Full deployment with vLLM endpoint
python scripts/export_merged_model.py \
    --checkpoint checkpoints/global_step_6 \
    --push_to_hub \
    --hub_repo username/model-name \
    --create_endpoint
```

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | - | SkyRL checkpoint directory |
| `--output` | `exports/merged_model` | Output path |
| `--base_model` | `Qwen/Qwen3-4B-Instruct-2507` | Base model |
| `--torch_dtype` | `bfloat16` | Model precision |
| `--verify` | false | Run verification test |
| `--push_to_hub` | false | Upload to HuggingFace |
| `--hub_repo` | `HF_REPO` env | Repository name |
| `--max_model_len` | 8192 | vLLM context length |
| `--create_endpoint` | false | Create inference endpoint |
| `--instance_type` | `nvidia-t4` | GPU type for endpoint |

### Checkpoint Structure

SkyRL saves LoRA adapters in PEFT format:

```
checkpoints/global_step_N/
├── policy/
│   ├── lora_adapter/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── model_state.pt (fallback)
```

### Environment Variables

```bash
export HF_TOKEN="..."  # HuggingFace API token
export HF_REPO="..."   # Default repository name (optional)
```

---

## 9. Results Analysis

Visualize and analyze evaluation results.

### Usage

```bash
# Generate chart and summary from default results
python scripts/analyze_results.py

# Custom input/output paths
python scripts/analyze_results.py \
    --results results/eval_results.json \
    --output results/eval_chart.png
```

### Output

1. **Bar Chart** (`results/eval_chart.png`):
   - Horizontal grouped bars by model
   - Metrics: relevance, clarity, discriminative, composite
   - Models sorted by composite score
   - Error bars showing standard deviation

2. **Console Summary**:
   - Mean ± std for each metric
   - Rankings by metric
   - Sample counts per model

---

## 10. Training Results

### Benchmark Summary

| Metric | Baseline | RL-Trained | Improvement |
|--------|----------|------------|-------------|
| Test Score | 0.795 | 0.863 | **+8.6%** |

### Score Progression

```
Step 0: 0.795 (baseline)
Step 2: 0.800
Step 4: 0.847
Step 5: 0.863 (final)
```

### Multi-Model Comparison (100 test samples)

Results from `results/eval_results.json`:

| Model | Composite | Notes |
|-------|-----------|-------|
| **RL-Trained** | 8.81 | Best performer |
| Grok 4.1 | 8.44 | SOTA API model |
| GPT-5 Mini | 8.25 | |
| Claude 4.5 Haiku | 8.21 | |
| Baseline | 6.44 | Pre-training |

### Training Configuration

- **Model**: Qwen3-4B-Instruct with LoRA (rank 8, alpha 16)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Data**: 400 training roles, 100 test roles
- **Duration**: ~19 minutes on single GPU

---

## 11. Verification & Debugging

### Expected Console Output

When training is working, you should see:

```
[Epoch 1] Step 10/100
  mean_reward: 0.42      # Should increase over time
  std_reward: 0.15
  kl_divergence: 0.002   # Should stay small (<0.1)

[Epoch 1] Step 20/100
  mean_reward: 0.48      # Increasing = model is learning
  std_reward: 0.12
```

### What Reward Values Mean

| Reward Range | Interpretation |
|--------------|----------------|
| 0.0 - 0.3 | Poor outputs, model still learning |
| 0.3 - 0.5 | Moderate quality |
| 0.5 - 0.7 | Good outputs |
| 0.7 - 1.0 | Excellent outputs |
| Constant 0.0 | Judge API failing (check logs) |
| Constant ~0.5 | No learning signal (check reward function) |

### Common Errors & Fixes

**"Environment 'my-gen-env' not found"**
```python
# Ensure register() is called before training starts
# Check the entry_point path matches your module structure
```

**"API rate limit exceeded"**
```python
# Add retry logic to reward.py
import time
for attempt in range(3):
    try:
        response = httpx.post(...)
        break
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            time.sleep(2 ** attempt)
```

**"CUDA out of memory"**
```yaml
# Reduce batch sizes in config
trainer:
  train_batch_size: 16  # Was 32
  micro_train_batch_size_per_gpu: 1  # Was 2
generator:
  n_samples_per_prompt: 2  # Was 4
```

**"Reward always 0"**
```python
# Debug your reward function
from src.recruiter.reward import judge_question

reward, details = judge_question(
    role_description="Backend engineer role",
    question="Explain how you would design a reliable REST pagination strategy."
)
print(f"Reward: {reward}, Details: {details}")
# Check if details contains an error
```

### Quick Sanity Check

Run this before full training:

```python
"""Test your setup before training."""
from src.recruiter.reward import judge_question
from src.recruiter.env import QuestionGenEnv
from src.recruiter.prompts import format_prompt

# Test reward function
print("Testing reward function...")
reward, details = judge_question(
    role_description="Backend engineer building REST APIs with Python and PostgreSQL.",
    question="How would you design pagination for a large list endpoint?"
)
print(f"  Reward: {reward:.2f}")
print(f"  Details: {details}")
assert reward >= 0, "Reward function failed"

# Test environment
print("\nTesting environment...")
role = {
    "id": "be-test",
    "title": "Backend Engineer",
    "level": "mid",
    "focus": "api-development",
    "stack": ["Python", "FastAPI", "PostgreSQL"],
    "description": "Build and maintain REST APIs.",
    "key_skills": ["Python", "API design", "SQL"]
}
prompt = format_prompt(role)
env = QuestionGenEnv()
env.init(prompt)
result = env.step("Design a rate limiting strategy for an API gateway.")
print(f"  Reward: {result.reward:.2f}")
print(f"  Done: {result.done}")
print(f"  Metadata keys: {list(result.metadata.keys())}")
assert result.done, "Environment should terminate after one step!"

print("\nAll checks passed!")
```

This snippet uses the same modules the trainer invokes (`src/recruiter/reward.py`, `env.py`, `prompts.py`) and assumes `XAI_API_KEY` is set. If it passes, `python -m src.recruiter.main` should work because `ensure_data_formatted()` will regenerate processed parquet files when needed.

---

## 12. What's Next

Once training is working:

### Production Enhancements
See the [full PRD](PRD.md) for:
- Multi-domain training
- Dataset preparation utilities
- Advanced configuration options

### Advanced Topics
- **Multi-turn**: Chain multiple generations with context
- **Custom criteria**: Add domain-specific scoring dimensions
- **Human feedback**: Mix judge scores with human preferences

---

## 13. Future Roadmap

### Human Feedback Integration

Incorporating human ratings from feedback-ui:
- Database schema ready (`feedback-ui/database.py`)
- Fields: question, role_context, relevance, clarity, discriminative (1-5 scale)
- Reward normalization: `(relevance + clarity + discriminative) / 15`
- **Status**: DB layer complete, FastAPI app not implemented

### Multi-turn Conversations

Extending from single-turn to multi-step generation:
- Current: One question per role
- Future: Follow-up questions based on candidate responses
- Requires environment changes (`done=False` for intermediate steps)

### Custom Judge Providers

Adding alternative judges beyond Grok:
- Claude judge (via Anthropic API)
- GPT judge (via OpenAI API)
- Ensemble judging (average multiple judges)

### Batch Evaluation Workflows

CI/CD integration for automated evaluation:
- GitHub Actions workflow for eval on PR
- Automated comparison against baseline
- Regression detection

---

## 14. Human Feedback UI (Optional / WIP)

Current repo state: only `feedback-ui/database.py` and `feedback-ui/requirements.txt` exist. The FastAPI UI and Grok-backed generator described in earlier drafts are **not implemented yet**—treat this as a starter DB layer.

### What Exists

```
feedback-ui/
├── database.py      # SQLAlchemy schema for feedback storage
├── requirements.txt # FastAPI, uvicorn, sqlalchemy, etc.
├── data/           # Empty (for SQLite DB)
├── static/         # Empty (for CSS/JS)
└── templates/      # Empty (for Jinja2 templates)
```

### Database Schema

| Field | Type | Description |
|-------|------|-------------|
| `question` | Text | Generated question |
| `role_context` | Text | Role description |
| `role_id` | String | Role identifier |
| `relevance` | Int (1-5) | Human rating |
| `clarity` | Int (1-5) | Human rating |
| `discriminative` | Int (1-5) | Human rating |
| `comments` | Text | Optional feedback |
| `source` | String | "generated" or "user_provided" |
| `created_at` | DateTime | Timestamp |

### To Complete This Feature

1. Create FastAPI app with routes for question submission and rating
2. Add Jinja2 templates with rating forms
3. Wire human feedback to training reward: `reward = (r + c + d) / 15`
4. Keep using `data/backend_roles.json` for role metadata consistency

## Quick Reference

### File Checklist

**Core Training**
- [ ] `src/recruiter/reward.py` - Grok judge with JSON sanitization (171 lines)
- [ ] `src/recruiter/env.py` - SkyRL environment wrapper (119 lines)
- [ ] `src/recruiter/main.py` - Training entrypoint with auto-formatting (104 lines)
- [ ] `src/recruiter/prompts.py` - Prompt formatting logic (51 lines)
- [ ] `src/recruiter/schemas.py` - Pydantic schemas for judge response
- [ ] `src/recruiter/dataset.py` - Alternative runtime formatting utility (54 lines)
- [ ] `configs/train_config.yaml` - Training configuration

**Data**
- [ ] `data/backend_roles.json` - Source role definitions (500 roles)
- [ ] `data/raw/train.parquet` & `test.parquet` - Raw role data
- [ ] `data/processed/train.parquet` & `test.parquet` - Formatted prompts (auto-generated)

**Evaluation & Export**
- [ ] `scripts/eval.py` - Multi-model evaluation (547 lines)
- [ ] `scripts/export_merged_model.py` - LoRA merge & Hub push (626 lines)
- [ ] `scripts/analyze_results.py` - Results visualization (356 lines)

**Utilities**
- [ ] `scripts/prepare_dataset.py` - JSON to parquet conversion
- [ ] `scripts/format_prompts.py` - Manual prompt formatting (optional)

### Environment Variables

```bash
# Required for training and evaluation
export XAI_API_KEY="your-grok-api-key"

# Optional: For multi-model evaluation
export ANTHROPIC_API_KEY="your-claude-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# Optional: For HuggingFace Hub push
export HF_TOKEN="your-huggingface-token"
export HF_REPO="username/model-name"
```

### Commands

```bash
# Training
python -m src.recruiter.main

# Evaluation
python scripts/eval.py --model baseline rl --checkpoint checkpoints/global_step_6

# Export
python scripts/export_merged_model.py --checkpoint checkpoints/global_step_6 --verify

# Analysis
python scripts/analyze_results.py
```

---

## License

MIT License
