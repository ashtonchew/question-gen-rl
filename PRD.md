# Technical Question Generator with Online RL

## Product Requirements Document

**Version:** 1.0
**Date:** December 2025
**Model:** grok-4-1-fast-non-reasoning

---

## Overview

This project implements an online reinforcement learning system that trains a language model to generate high-quality technical screening questions for recruiting. The system uses **grok-4-1-fast-non-reasoning** as both the policy model being trained and the reward judge.

### Core Concept

- **Input:** Role descriptions (title, level, domain, skills)
- **Output:** Technical screening questions tailored to each role
- **Training:** Online RL using GRPO (Group Relative Policy Optimization)
- **Reward:** LLM-as-judge scoring on relevance, clarity, and discriminative power

---

## Project Structure

```
question-gen-rl/
├── README.md
├── PRD.md                         # This document
├── pyproject.toml
├── setup.sh                       # Bootstrap script
├── data/
│   ├── backend_roles.json         # 100 backend roles (raw)
│   └── processed/
│       ├── train.parquet          # 80 for training
│       └── test.parquet           # 20 for eval (held out)
├── scripts/
│   ├── prepare_dataset.py         # Split into train/test
│   ├── train.sh                   # Run RL training
│   └── eval.py                    # Compare all models
├── src/
│   └── recruiter/
│       ├── __init__.py
│       ├── env.py                 # SkyRL environment
│       ├── reward.py              # Grok API judge
│       └── main.py                # Training entrypoint
├── configs/
│   └── train_config.yaml          # Hydra config overrides
└── results/                       # Eval outputs go here
```

---

## 1. Dataset: Backend Roles Only

### File: `data/backend_roles.json`

100 backend engineering roles with variation in:
- **Level:** junior, mid, senior, staff, principal
- **Focus area:** APIs, databases, distributed systems, performance, security, DevOps
- **Tech stack:** Python, Go, Java, Node, Rust, etc.

```json
[
  {
    "id": "be-001",
    "title": "Senior Backend Engineer",
    "level": "senior",
    "focus": "distributed-systems",
    "stack": ["Go", "Kubernetes", "PostgreSQL", "Kafka"],
    "description": "Design and build distributed systems handling 100k+ RPS. Own service reliability, data consistency, and cross-team API contracts.",
    "key_skills": ["distributed systems", "Go", "system design", "reliability"]
  },
  {
    "id": "be-002",
    "title": "Backend Engineer",
    "level": "mid",
    "focus": "api-development",
    "stack": ["Python", "FastAPI", "PostgreSQL", "Redis"],
    "description": "Build and maintain REST APIs for our core product. Focus on clean abstractions, test coverage, and performance optimization.",
    "key_skills": ["Python", "API design", "SQL", "testing"]
  }
]
```

### Scope Recommendation

Pick **ONE domain** to start (e.g., backend). Expand after validating the approach.

---

## Why Single Domain is Better

1. **Small model + narrow domain = cleaner signal** — Qwen3-4B learning to generate great backend questions is achievable. Learning all domains is not.
2. **Actually measurable** — You can show "our RL model beats baseline on backend questions" with statistical significance
3. **Hackathon-scoped** — One domain done well > five domains done poorly

---

## Revised Spec: Backend Engineering Question Generator

### Dataset Structure

```
data/
├── backend_roles.json      # 100 role descriptions (backend only)
├── train.parquet           # 80 roles for training
├── test.parquet            # 20 roles held out for eval
```

### Evaluation Plan

| Model | What we measure |
|-------|-----------------|
| **Qwen3-4B base** (no RL) | Baseline — just prompt it to generate questions |
| **Qwen3-4B + RL** (ours) | After training |
| **Grok-4-1-fast / Claude haiku 4.5** (SOTA) | Upper bound — what's possible with a frontier model |

**Metrics (from the LLM judge):**
- Relevance (0-10)
- Clarity (0-10)
- Discriminative power (0-10)
- **Composite score** = average of all three

### What You Demo

```
"On held-out backend roles:
- Qwen3-4B baseline: 5.2 avg score
- Qwen3-4B + RL (ours): 7.1 avg score (+36%)
- Grok-3 (SOTA): 8.3 avg score

We closed 34% of the gap to SOTA with online RL."
```

---

## 2. Reward Function: Grok API Judge

### File: `src/recruiter/reward.py`

Uses **grok-4-1-fast-non-reasoning** for reward scoring.

```python
import os
import httpx
from typing import Tuple

GROK_API_KEY = os.environ.get("XAI_API_KEY")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

JUDGE_SYSTEM_PROMPT = """You are an expert technical recruiter evaluating screening questions.

Score the following technical screening question on three criteria (0-10 each):

1. **Relevance** (0-10): Does this question test skills actually needed for the role?
   - 0: Completely unrelated to the role
   - 5: Tangentially related
   - 10: Directly tests core job requirements

2. **Clarity** (0-10): Is the question unambiguous and well-formed?
   - 0: Confusing, multiple interpretations possible
   - 5: Understandable but could be clearer
   - 10: Crystal clear, one obvious interpretation

3. **Discriminative Power** (0-10): Would this question distinguish good candidates from weak ones?
   - 0: Anyone could answer OR no one could answer
   - 5: Moderate differentiation
   - 10: Strong candidates would excel, weak candidates would struggle

Respond with ONLY a JSON object:
{"relevance": <int>, "clarity": <int>, "discriminative": <int>, "reasoning": "<brief explanation>"}
"""


def judge_question(role_description: str, question: str) -> Tuple[float, dict]:
    """
    Call Grok API to score a generated question.
    Returns (normalized_reward, details_dict)
    """
    user_prompt = f"""## Role Description
{role_description}

## Generated Screening Question
{question}

Score this question:"""

    response = httpx.post(
        GROK_API_URL,
        headers={
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "grok-4-1-fast-non-reasoning",
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0  # Deterministic scoring
        },
        timeout=30.0
    )

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from response
    import json
    try:
        scores = json.loads(content)
        relevance = scores.get("relevance", 0)
        clarity = scores.get("clarity", 0)
        discriminative = scores.get("discriminative", 0)

        # Normalize to 0-1 range, weighted average
        reward = (relevance + clarity + discriminative) / 30.0

        return reward, scores
    except json.JSONDecodeError:
        # Fallback if parsing fails
        return 0.0, {"error": "Failed to parse judge response", "raw": content}
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Judge Model | grok-4-1-fast-non-reasoning | Fast reasoning for quick, accurate scoring |
| Temperature | 0.0 | Deterministic rewards for stable training |
| Timeout | 30s | Allow for reasoning time |
| Scoring | 3 criteria, normalized 0-1 | Balanced multi-objective reward |

---

## 3. SkyRL Environment

### File: `src/recruiter/env.py`

```python
from dataclasses import dataclass
from typing import Optional
import random

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .reward import judge_question


@dataclass
class QuestionGenEnvConfig:
    """Configuration for the question generation environment."""
    max_question_length: int = 500
    min_question_length: int = 20


class QuestionGenEnv(BaseTextEnv):
    """
    Environment for training a technical question generator.

    Single-turn environment:
    - State: Role description prompt
    - Action: Model generates a screening question
    - Reward: LLM judge score (0-1)
    - Done: Always True after one generation
    """

    def __init__(self, config: Optional[QuestionGenEnvConfig] = None):
        super().__init__()
        self.config = config or QuestionGenEnvConfig()
        self.current_role: Optional[dict] = None
        self._initial_prompt: Optional[str] = None

    def set_role(self, role: dict):
        """Set the current role for this episode."""
        self.current_role = role
        self._initial_prompt = self._format_prompt(role)

    def _format_prompt(self, role: dict) -> str:
        """Format the role into a prompt for the model."""
        return f"""You are a technical recruiter creating screening questions.

## Role: {role['title']}
**Level:** {role['level']}
**Domain:** {role['domain']}

**Description:** {role['description']}

**Key Skills:** {', '.join(role['key_skills'])}

---

Generate ONE technical screening question for this role. The question should:
- Be answerable in 2-5 minutes
- Test practical knowledge, not trivia
- Be appropriate for the seniority level

Question:"""

    def init(self) -> str:
        """Return the initial prompt for the model."""
        if self._initial_prompt is None:
            raise ValueError("Must call set_role() before init()")
        return self._initial_prompt

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process the model's generated question.

        Args:
            action: The generated question from the model

        Returns:
            BaseTextEnvStepOutput with reward and done=True
        """
        question = action.strip()

        # Basic validation
        if len(question) < self.config.min_question_length:
            return BaseTextEnvStepOutput(
                observation="",  # No further observation needed
                reward=-0.5,     # Penalty for too-short output
                terminated=True,
                truncated=False,
                info={"error": "Question too short"}
            )

        if len(question) > self.config.max_question_length:
            question = question[:self.config.max_question_length]

        # Get reward from judge
        reward, judge_details = judge_question(
            role_description=self.current_role['description'],
            question=question
        )

        return BaseTextEnvStepOutput(
            observation="",
            reward=reward,
            terminated=True,  # Single-turn: always done
            truncated=False,
            info={
                "question": question,
                "judge_scores": judge_details,
                "role_id": self.current_role['id']
            }
        )
```

### Environment Characteristics

| Property | Value | Notes |
|----------|-------|-------|
| Episode Length | 1 turn | Single generation per role |
| State Space | Text prompt | Role description formatted |
| Action Space | Text | Generated question |
| Reward Range | [-0.5, 1.0] | Penalty for invalid, 0-1 for scored |

---

## 4. Dataset Preparation Script

### File: `scripts/prepare_dataset.py`

```python
"""Convert backend_roles.json to SkyRL parquet format."""
import json
import pandas as pd
from pathlib import Path
import argparse


def format_prompt(role: dict) -> str:
    """Format role into training prompt."""
    return f"""You are a technical recruiter creating screening questions.

## Role: {role['title']}
**Level:** {role['level']}
**Domain:** {role['domain']}

**Description:** {role['description']}

**Key Skills:** {', '.join(role['key_skills'])}

---

Generate ONE technical screening question for this role. The question should:
- Be answerable in 2-5 minutes
- Test practical knowledge, not trivia
- Be appropriate for the seniority level

Question:"""


def main(input_path: str, output_dir: str, train_split: float = 0.8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load roles
    with open(input_path) as f:
        roles = json.load(f)

    # Convert to SkyRL format
    # SkyRL expects: prompt (str) and optionally other metadata
    records = []
    for role in roles:
        records.append({
            "prompt": format_prompt(role),
            "role_id": role["id"],
            "role_title": role["title"],
            "role_level": role["level"],
            "role_domain": role["domain"],
            # Store full role as JSON string for env access
            "role_json": json.dumps(role)
        })

    df = pd.DataFrame(records)

    # Split train/val
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Save as parquet
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Created {len(train_df)} training examples")
    print(f"Created {len(val_df)} test examples")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/backend_roles.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    main(args.input, args.output_dir, args.train_split)
```

---

## 5. Evaluation Script

### File: `scripts/eval.py`

Run on test set, compare models:

```python
"""Evaluate question generation across models."""
import json
from pathlib import Path
from statistics import mean, stdev

from src.recruiter.reward import judge_question

def generate_question_baseline(role: dict, model: str = "Qwen/Qwen3-4B") -> str:
    """Generate question with base model (no RL)."""
    # Call vLLM or transformers directly
    ...

def generate_question_rl(role: dict, checkpoint_path: str) -> str:
    """Generate question with RL-trained model."""
    ...

def generate_question_sota(role: dict) -> str:
    """Generate question with Grok-3 as upper bound."""
    ...

def evaluate_model(roles: list[dict], generate_fn) -> dict:
    """Evaluate a model on all test roles."""
    scores = []
    for role in roles:
        question = generate_fn(role)
        reward, details = judge_question(role["description"], question)
        scores.append({
            "role_id": role["id"],
            "question": question,
            "relevance": details.get("relevance", 0),
            "clarity": details.get("clarity", 0),
            "discriminative": details.get("discriminative", 0),
            "composite": reward * 30  # Scale back to 0-30
        })

    composite_scores = [s["composite"] for s in scores]
    return {
        "mean": mean(composite_scores),
        "std": stdev(composite_scores),
        "scores": scores
    }

def main():
    # Load test set
    test_roles = json.load(open("data/backend_roles.json"))

    results = {
        "baseline": evaluate_model(test_roles, generate_question_baseline),
        "rl": evaluate_model(test_roles, generate_question_rl),
        "sota": evaluate_model(test_roles, generate_question_sota),
    }

    print(f"Baseline (Qwen3-4B):     {results['baseline']['mean']:.1f} ± {results['baseline']['std']:.1f}")
    print(f"RL (Ours):               {results['rl']['mean']:.1f} ± {results['rl']['std']:.1f}")
    print(f"SOTA (Grok-3):           {results['sota']['mean']:.1f} ± {results['sota']['std']:.1f}")

    # Save full results
    json.dump(results, open("results/eval_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

---

## 6. Training Entrypoint

### File: `src/recruiter/main.py`

```python
"""Main entrypoint for training the question generator."""
import ray
import hydra
from omegaconf import DictConfig
import json

from skyrl_train.experiments.base_ppo_exp import BasePPOExp
from skyrl_train.utils.config import validate_cfg
from skyrl_train.utils.ray_utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


class QuestionGenEnvWrapper(QuestionGenEnv):
    """
    Wrapper that initializes from the dataset row.
    SkyRL passes the dataset row to the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(QuestionGenEnvConfig())
        self._role_json = kwargs.get("role_json", None)
        if self._role_json:
            role = json.loads(self._role_json)
            self.set_role(role)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register our custom environment
    register(
        id="question-gen",
        entry_point="src.recruiter.main:QuestionGenEnvWrapper",
    )

    # Run training
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

---

## 7. Training Configuration

### File: `configs/train_config.yaml`

```yaml
# Hydra config for question generation training with grok-4-1-fast-non-reasoning

defaults:
  - _self_

# Data paths (override on CLI)
data:
  train_data: ["data/processed/train.parquet"]
  val_data: ["data/processed/test.parquet"]

# Model config
trainer:
  policy:
    model:
      path: "Qwen/Qwen3-4B-Instruct"  # Small model for 1 GPU

  # Algorithm
  algorithm:
    advantage_estimator: "grpo"

  # Placement for 1 GPU
  placement:
    colocate_all: true
    policy_num_gpus_per_node: 1
    ref_num_gpus_per_node: 1
    policy_num_nodes: 1
    ref_num_nodes: 1

  strategy: fsdp2

  # Training params
  epochs: 20
  train_batch_size: 64        # Smaller for 1 GPU
  eval_batch_size: 32
  micro_train_batch_size_per_gpu: 4
  micro_forward_batch_size_per_gpu: 4
  eval_before_train: true
  eval_interval: 5
  update_epochs_per_batch: 1

  # Logging
  logger: "console"  # or "wandb"

# Generator config
generator:
  num_inference_engines: 1
  inference_engine_tensor_parallel_size: 1
  max_new_tokens: 256         # Questions shouldn't be super long
  batched: true               # Single-turn, can batch
  async_engine: false

# Environment config
env:
  id: "question-gen"
```

---

## 8. Project Configuration

### File: `pyproject.toml`

```toml
[project]
name = "question-gen-rl"
version = "0.1.0"
description = "Online RL for technical question generation using grok-4-1-fast-non-reasoning"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
]

[project.optional-dependencies]
train = [
    "skyrl-train",
    "skyrl-gym",
    "torch>=2.0.0",
    "vllm>=0.4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 9. Bootstrap Script

### File: `setup.sh`

```bash
#!/bin/bash
set -e

echo "=== Question Gen RL Setup ==="

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone SkyRL (needed as dependency)
cd ~
if [ ! -d "SkyRL" ]; then
    git clone https://github.com/NovaSky-AI/SkyRL.git
fi

# Setup your project
cd ~/Desktop/question-gen-rl
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install SkyRL from local clone
uv pip install -e ~/SkyRL/skyrl-train[vllm]
uv pip install -e ~/SkyRL/skyrl-gym
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
```

---

## 10. Implementation Checklist

### Phase 1: Project Setup
- [ ] Create directory structure
- [ ] Initialize `pyproject.toml`
- [ ] Create `src/recruiter/__init__.py`
- [ ] Write `setup.sh` bootstrap script

### Phase 2: Data Layer
- [ ] Create initial `data/backend_roles.json` with 100 backend roles
- [ ] Implement `scripts/prepare_dataset.py`
- [ ] Generate parquet files

### Phase 3: Core Components
- [ ] Implement `src/recruiter/reward.py` with grok-4-1-fast-non-reasoning
- [ ] Implement `src/recruiter/env.py` SkyRL environment
- [ ] Implement `src/recruiter/main.py` training entrypoint

### Phase 4: Configuration
- [ ] Create `configs/train_config.yaml`
- [ ] Test configuration loading

### Phase 5: Integration Testing
- [ ] Test reward function with sample questions
- [ ] Test environment step/reset cycle
- [ ] Run single training iteration

### Phase 6: Training
- [ ] Full training run on Lambda/cloud GPU
- [ ] Monitor rewards and generate samples
- [ ] Iterate on reward function if needed

---

## 11. API Reference

### Grok API (x.ai)

**Endpoint:** `https://api.x.ai/v1/chat/completions`

**Model:** `grok-4-1-fast-non-reasoning`

**Authentication:** Bearer token via `XAI_API_KEY` environment variable

**Request Format:**
```json
{
  "model": "grok-4-1-fast-non-reasoning",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.0
}
```

**Get API Key:** https://console.x.ai

---

## 12. Evaluation Metrics

### Training Metrics
- **Mean Reward:** Average judge score per batch
- **Reward Variance:** Stability of generation quality
- **Episode Length:** Should always be 1 (single-turn)

### Quality Metrics (offline evaluation)
- **Relevance Score:** Average relevance across test roles
- **Clarity Score:** Average clarity across test roles
- **Discriminative Score:** Average discriminative power
- **Human Preference:** A/B test vs baseline questions

---

## 13. Future Enhancements

1. **Multi-turn Generation:** Follow-up questions based on initial response
2. **Difficulty Calibration:** Adjust question difficulty to role level
3. **Domain Expansion:** Add more domains beyond initial focus
4. **Answer Key Generation:** Generate model answers for each question
5. **Candidate Simulation:** Simulate candidate responses for evaluation
6. **Fine-tuned Judge:** Train specialized judge model on human preferences

---

## 14. Realistic 24-Hour Timeline

| Hours | Task |
|-------|------|
| 0-2 | Setup Lambda, clone SkyRL, verify it runs |
| 2-4 | Generate 100 backend roles (use Claude/Grok to help), prepare dataset |
| 4-6 | Implement reward function, test Grok API judge works |
| 6-8 | Implement SkyRL env, debug until training loop runs |
| 8-16 | **Training** — run RL, monitor, iterate on hyperparams |
| 16-20 | Run eval script, collect baseline + SOTA comparisons |
| 20-24 | Polish demo, write up results, prepare presentation |

---

## Quick Start

```bash
# 1. Clone and setup
cd ~/Desktop/question-gen-rl
chmod +x setup.sh
./setup.sh

# 2. Set API key
export XAI_API_KEY="your-key-here"

# 3. Run training
source .venv/bin/activate
python -m src.recruiter.main
```

---

## License

MIT License
