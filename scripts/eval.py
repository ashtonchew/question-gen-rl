"""Evaluation script for comparing baseline, RL-trained, and SOTA models."""
import json
import os
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import httpx
from xai_sdk import Client
from xai_sdk.chat import system, user
from src.recruiter.schemas import JudgeResponse

# API configuration
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Initialize xai-sdk client with 10 minute timeout
# SDK has built-in retry with exponential backoff
xai_client = Client(api_key=XAI_API_KEY, timeout=600) if XAI_API_KEY else None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def format_prompt(role: dict) -> str:
    """Format role into generation prompt."""
    focus = role.get('focus', role.get('domain', 'backend'))
    stack = role.get('stack', [])
    stack_str = ', '.join(stack) if stack else 'Not specified'

    return f"""You are a technical recruiter creating screening questions.

## Role: {role['title']}
**Level:** {role['level']}
**Focus Area:** {focus}
**Tech Stack:** {stack_str}

**Description:** {role['description']}

**Key Skills:** {', '.join(role['key_skills'])}

---

Generate ONE technical screening question for this role. The question should:
- Be answerable in 2-5 minutes
- Test practical knowledge, not trivia
- Be appropriate for the seniority level

Question:"""


def generate_question_baseline(role: dict, model_path: str = "Qwen/Qwen3-4B-Instruct-2507") -> str:
    """
    Generate question using baseline model (no RL training).
    Uses vLLM for local inference.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    # Initialize model (cached after first call)
    if not hasattr(generate_question_baseline, "_llm"):
        generate_question_baseline._llm = LLM(model=model_path)

    llm = generate_question_baseline._llm
    prompt = format_prompt(role)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        stop=["\n\n", "---"]
    )

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_question_rl(role: dict, checkpoint_path: str) -> str:
    """
    Generate question using RL-trained model.
    Loads model from checkpoint.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    # Initialize model from checkpoint
    if not hasattr(generate_question_rl, "_llm") or generate_question_rl._checkpoint != checkpoint_path:
        generate_question_rl._llm = LLM(model=checkpoint_path)
        generate_question_rl._checkpoint = checkpoint_path

    llm = generate_question_rl._llm
    prompt = format_prompt(role)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        stop=["\n\n", "---"]
    )

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_question_sota(role: dict, provider: str = "grok") -> str:
    """
    Generate question using SOTA model (Grok or Claude).
    Uses xai-sdk for Grok with built-in retry and 10min timeout.
    """
    prompt = format_prompt(role)

    if provider == "grok":
        if xai_client is None:
            raise ValueError("XAI_API_KEY environment variable not set")

        chat = xai_client.chat.create(model="grok-4-1-fast-non-reasoning")
        chat.append(user(prompt))
        response = chat.sample()
        return response.content.strip()

    elif provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        response = httpx.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60.0
        )
        result = response.json()
        return result["content"][0]["text"].strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


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
"""


def judge_question(role_description: str, question: str) -> dict:
    """
    Score a question using Grok as judge.
    Uses xai-sdk structured outputs for guaranteed schema compliance.
    Returns dict with relevance, clarity, discriminative, and composite scores.
    """
    if xai_client is None:
        raise ValueError("XAI_API_KEY environment variable not set")

    user_prompt = f"""## Role Description
{role_description}

## Generated Screening Question
{question}

Score this question:"""

    chat = xai_client.chat.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[system(JUDGE_SYSTEM_PROMPT)]
    )
    chat.append(user(user_prompt))

    # Use structured outputs - guaranteed schema compliance
    response, scores = chat.parse(JudgeResponse)

    result = scores.model_dump()
    result["composite"] = (scores.relevance + scores.clarity + scores.discriminative) / 3.0
    return result


def evaluate_model(
    test_data_path: str,
    model_type: str,
    checkpoint_path: Optional[str] = None,
    sota_provider: str = "grok",
    num_samples: Optional[int] = None
) -> dict:
    """
    Evaluate a model on test data.

    Args:
        test_data_path: Path to test.parquet
        model_type: One of 'baseline', 'rl', 'sota'
        checkpoint_path: Required if model_type is 'rl'
        sota_provider: 'grok' or 'claude' if model_type is 'sota'
        num_samples: Number of samples to evaluate (None for all)

    Returns:
        Dict with aggregate metrics and per-sample results
    """
    # Load test data
    df = pd.read_parquet(test_data_path)

    if num_samples:
        df = df.head(num_samples)

    results = []
    total_relevance = 0
    total_clarity = 0
    total_discriminative = 0
    total_composite = 0

    for idx, row in df.iterrows():
        role = json.loads(row["role_json"])

        # Generate question based on model type
        if model_type == "baseline":
            question = generate_question_baseline(role)
        elif model_type == "rl":
            if not checkpoint_path:
                raise ValueError("checkpoint_path required for RL model")
            question = generate_question_rl(role, checkpoint_path)
        elif model_type == "sota":
            question = generate_question_sota(role, sota_provider)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Score the question
        scores = judge_question(role["description"], question)

        result = {
            "role_id": role["id"],
            "role_title": role["title"],
            "question": question,
            "scores": scores
        }
        results.append(result)

        # Accumulate scores
        total_relevance += scores.get("relevance", 0)
        total_clarity += scores.get("clarity", 0)
        total_discriminative += scores.get("discriminative", 0)
        total_composite += scores.get("composite", 0)

        print(f"[{idx+1}/{len(df)}] {role['title']}: {scores.get('composite', 0):.2f}")

    n = len(results)
    return {
        "model_type": model_type,
        "num_samples": n,
        "avg_relevance": total_relevance / n if n > 0 else 0,
        "avg_clarity": total_clarity / n if n > 0 else 0,
        "avg_discriminative": total_discriminative / n if n > 0 else 0,
        "avg_composite": total_composite / n if n > 0 else 0,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate question generation models")
    parser.add_argument("--test_data", default="data/processed/test.parquet",
                        help="Path to test parquet file")
    parser.add_argument("--model", choices=["baseline", "rl", "sota", "all"],
                        default="all", help="Model to evaluate")
    parser.add_argument("--checkpoint", help="Path to RL checkpoint (required for --model=rl)")
    parser.add_argument("--sota_provider", choices=["grok", "claude"],
                        default="grok", help="SOTA model provider")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--output", default="results/eval_results.json",
                        help="Output path for results")
    args = parser.parse_args()

    # Create results directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.model in ["baseline", "all"]:
        print("\n=== Evaluating Baseline Model ===")
        all_results["baseline"] = evaluate_model(
            args.test_data,
            "baseline",
            num_samples=args.num_samples
        )

    if args.model in ["rl", "all"]:
        if not args.checkpoint:
            if args.model == "rl":
                raise ValueError("--checkpoint required for RL evaluation")
            print("\n=== Skipping RL Model (no checkpoint provided) ===")
        else:
            print("\n=== Evaluating RL Model ===")
            all_results["rl"] = evaluate_model(
                args.test_data,
                "rl",
                checkpoint_path=args.checkpoint,
                num_samples=args.num_samples
            )

    if args.model in ["sota", "all"]:
        print(f"\n=== Evaluating SOTA Model ({args.sota_provider}) ===")
        all_results["sota"] = evaluate_model(
            args.test_data,
            "sota",
            sota_provider=args.sota_provider,
            num_samples=args.num_samples
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for model_name, result in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Samples: {result['num_samples']}")
        print(f"  Avg Relevance: {result['avg_relevance']:.2f}")
        print(f"  Avg Clarity: {result['avg_clarity']:.2f}")
        print(f"  Avg Discriminative: {result['avg_discriminative']:.2f}")
        print(f"  Avg Composite: {result['avg_composite']:.2f}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
