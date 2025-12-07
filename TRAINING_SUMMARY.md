# Training Summary

## What This Is
An RL-trained model that generates technical screening questions for backend engineering roles.

**Input**: Role description (title, level, tech stack, focus area)
**Output**: A single screening question appropriate for that role

## Training Setup
- **Model**: Qwen3-4B-Instruct with LoRA (rank 8)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward**: LLM-as-judge (Grok API) scoring relevance, clarity, and discriminative power
- **Data**: 80 training roles, 20 test roles
- **Duration**: ~19 minutes on single GPU

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Test Score | 0.795 | 0.863 | **+8.6%** |

### Score Progression
```
Step 0: 0.795 (baseline)
Step 2: 0.800
Step 4: 0.847
Step 5: 0.863 (final)
```

## Example Output

**Role**: Senior Backend Engineer - Content Moderation (FastAPI, Redis, PostgreSQL)

**Generated Question**:
> Design a FastAPI endpoint that accepts user-uploaded content and returns a moderation decision. The system should store content in a Redis queue for async processing, use a rule-based filter to flag prohibited terms, and be rate-limited to 100 req/min. How would you extend this to integrate with an ML model later?

## Artifacts
- Checkpoint: `checkpoints/global_step_6/`
- LoRA Adapter: `checkpoints/global_step_6/policy/lora_adapter`
