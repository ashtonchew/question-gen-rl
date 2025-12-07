# Online RL Add-on Guide (MVP)

**Version:** 1.0
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
| `reward.py` | Calls judge API, returns normalized score | ~50 |
| `env.py` | Wraps your generation in SkyRL interface | ~50 |
| `config.yaml` | Essential training parameters | ~40 |

---

## 4. Core Components

### 4.1 Reward Function (`src/recruiter/reward.py`)

The repo already ships a Grok-based judge that uses `xai-sdk` structured outputs and retries. Use it as-is unless you need to change scoring dimensions.

```python
"""Reward function using Grok API as judge."""
import os
import logging
from typing import Tuple
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system, user
from src.recruiter.schemas import JudgeResponse

load_dotenv()
logger = logging.getLogger(__name__)

_api_key = os.environ.get("XAI_API_KEY")
client = Client(api_key=_api_key, timeout=600) if _api_key else None

JUDGE_SYSTEM_PROMPT = """You are an expert technical recruiter evaluating screening questions..."""

def judge_question(role_description: str, question: str) -> Tuple[float, dict]:
    if client is None:
        return 0.0, {"error": "XAI_API_KEY not set"}

    chat = client.chat.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[system(JUDGE_SYSTEM_PROMPT)],
        response_format="json_object"  # Force JSON (no markdown)
    )
    chat.append(user(f"## Role Description\n{role_description}\n\n## Generated Screening Question\n{question}\n\nScore this question:"))

    raw = chat.sample()
    scores = JudgeResponse.model_validate_json(raw.content)  # Strict schema validation
    reward = (scores.relevance + scores.clarity + scores.discriminative) / 30.0
    return reward, scores.model_dump()
```

Env var: `XAI_API_KEY` (required). Model/URL are hardcoded in code; change there if needed.

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

### 4.3 Training Entrypoint (`src/recruiter/main.py`)

```python
"""Training entrypoint."""
import ray
import hydra
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    register(
        id="question-gen",
        entry_point="src.recruiter.main:QuestionGenEnvWrapper",
    )
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
```

Run with:

```bash
python -m src.recruiter.main
```

---

## 5. Essential Configuration (mirrors `configs/train_config.yaml`)

Key defaults already present in the repo:

```yaml
data:
  train_data: ["data/processed/train.parquet"]
  val_data: ["data/processed/test.parquet"]

trainer:
  placement:
    colocate_all: true
    policy_num_nodes: 1
    policy_num_gpus_per_node: 1
  strategy: fsdp2
  sequence_parallel_backend: "ulysses"
  policy:
    model:
      path: "Qwen/Qwen3-4B-Instruct-2507"
      lora:
        rank: 1
    optimizer_config:
      lr: 1.0e-5
  algorithm:
    advantage_estimator: "grpo"
    kl_loss_coef: 0.001
  epochs: 5
  train_batch_size: 64
  micro_train_batch_size_per_gpu: 2
  eval_before_train: true
  eval_interval: 2
  ckpt_path: "checkpoints/"
  project_name: "question-gen-rl"
  run_name: "qwen3-4b-grpo"
  logger: "console"

generator:
  model_name: ${trainer.policy.model.path}
  backend: "vllm"
  n_samples_per_prompt: 4
  sampling_params:
    max_generate_length: 256
    temperature: 1.0

environment:
  env_class: "question-gen"   # must match register() id
  skyrl_gym:
    max_env_workers: 8
```

Hardware note: With fsdp2 + Qwen 4B + `n_samples_per_prompt=4`, expect ≥24 GB VRAM or CPU offload. To downscale, lower `train_batch_size`, `micro_train_batch_size_per_gpu`, and `n_samples_per_prompt`; reduce `skyrl_gym.max_env_workers` if memory is tight.

---

## 6. Integration Steps

### Step 1: Prepare Your Data (matches repo flow)

```bash
# 1) JSON → raw parquet
python scripts/prepare_dataset.py           # writes data/raw/train.parquet & test.parquet

# 2) Raw → formatted prompts (optional; auto-runs on training start)
python scripts/format_prompts.py            # writes data/processed/*.parquet
```

Dataset schema: raw parquet contains `role_json`; prompt formatting converts it to a two-message chat list (`[{"role": "system", ...}, {"role": "user", ...}]`) with `**ID:** <role_id>` embedded. The environment relies on that to extract the role id.

### Step 2: Set Environment Variable

```bash
export XAI_API_KEY="your-grok-api-key"
```

### Step 3: Run Training

```bash
python -m src.recruiter.main
```

---

## 7. Verification & Debugging

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

---

## 8. What's Next

Once training is working:

### Evaluate Against Baseline
Compare your RL-trained model against the base model:
- Generate outputs from both on held-out test set
- Score both with the same judge
- Measure improvement %

### Production Enhancements
See the [full PRD](PRD.md) for:
- Comprehensive evaluation scripts
- Multi-domain training
- SOTA model comparison
- Dataset preparation utilities

### Advanced Topics
- **Multi-turn**: Chain multiple generations with context
- **Custom criteria**: Add domain-specific scoring dimensions
- **Human feedback**: Mix judge scores with human preferences

---

## 9. Human Feedback UI (Optional / WIP)

Current repo state: only `feedback-ui/database.py` and `feedback-ui/requirements.txt` exist. The FastAPI UI and Grok-backed generator described in earlier drafts are **not implemented yet**.

If you continue this work:
- Reuse the existing schema in `feedback-ui/database.py` (fields: question, role_context, role_id, relevance, clarity, discriminative, comments, source, created_at).
- Keep using `data/backend_roles.json` for role metadata so training and UI stay consistent.
- Add a small FastAPI app plus templates/static; the dependencies are already listed in `feedback-ui/requirements.txt`.
- When wiring training to human feedback, look up the latest feedback for a question and normalize to a 0–1 reward as `(relevance + clarity + discriminative) / 15`.

## Quick Reference

### File Checklist
- [ ] `src/recruiter/reward.py` - Grok judge wrapper
- [ ] `src/recruiter/env.py` - SkyRL environment
- [ ] `src/recruiter/main.py` - Training entrypoint
- [ ] `configs/train_config.yaml` - Training config
- [ ] `data/raw/train.parquet` & `data/raw/test.parquet` - Raw data
- [ ] `data/processed/train.parquet` & `data/processed/test.parquet` - Formatted prompts

### Environment Variables
```bash
export XAI_API_KEY="your-grok-api-key"
```

### Launch Command
```bash
python -m src.recruiter.main
```

---

## License

MIT License
