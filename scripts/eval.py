"""Evaluation script for comparing models on question generation quality."""
import json
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import httpx
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system, user
from src.recruiter.schemas import JudgeResponse

# Load environment variables from .env file
load_dotenv()

# API configuration
XAI_API_KEY = os.environ.get("XAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize xai-sdk client with 10 minute timeout
# SDK has built-in retry with exponential backoff
xai_client = Client(api_key=XAI_API_KEY, timeout=600) if XAI_API_KEY else None

# Model aliases mapping short names to full model IDs
MODEL_ALIASES = {
    # SOTA models
    "grok-4-1": {"model": "grok-4-1-fast-non-reasoning", "provider": "xai"},
    "claude-4-5-haiku": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    "gpt-5-nano": {"model": "gpt-5-nano-2025-08-07", "provider": "openai"},
    # Local models
    "baseline": {"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "local"},
    "rl": {"model": None, "provider": "local"},  # Requires --checkpoint
}

# Judge model is hardcoded for fair comparison
JUDGE_MODEL = "grok-4-1-fast-non-reasoning"


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


def generate_question_local(role: dict, model_path: str) -> str:
    """Generate question using local model via vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    # Initialize model (cached after first call per model path)
    cache_key = f"_llm_{model_path}"
    if not hasattr(generate_question_local, cache_key):
        setattr(generate_question_local, cache_key, LLM(model=model_path))

    llm = getattr(generate_question_local, cache_key)
    prompt = format_prompt(role)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        stop=["\n\n", "---"]
    )

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_question_xai(role: dict, model: str) -> str:
    """Generate question using xAI API (Grok)."""
    if xai_client is None:
        raise ValueError("XAI_API_KEY environment variable not set")

    prompt = format_prompt(role)
    chat = xai_client.chat.create(model=model)
    chat.append(user(prompt))
    response = chat.sample()
    return response.content.strip()


def generate_question_anthropic(role: dict, model: str) -> str:
    """Generate question using Anthropic API (Claude)."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    prompt = format_prompt(role)
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        json={
            "model": model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60.0
    )
    result = response.json()
    return result["content"][0]["text"].strip()


def generate_question_openai(role: dict, model: str) -> str:
    """Generate question using OpenAI Responses API (GPT-5 series)."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = format_prompt(role)
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"}
    )

    return response.output_text.strip()


def generate_question(role: dict, model_alias: str, checkpoint_path: Optional[str] = None) -> str:
    """Generate question using the specified model."""
    if model_alias not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {model_alias}. Available: {list(MODEL_ALIASES.keys())}")

    config = MODEL_ALIASES[model_alias]
    provider = config["provider"]
    model = config["model"]

    if provider == "local":
        if model_alias == "rl":
            if not checkpoint_path:
                raise ValueError("--checkpoint required for RL model")
            model = checkpoint_path
        return generate_question_local(role, model)
    elif provider == "xai":
        return generate_question_xai(role, model)
    elif provider == "anthropic":
        return generate_question_anthropic(role, model)
    elif provider == "openai":
        return generate_question_openai(role, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


JUDGE_SYSTEM_PROMPT = """You are an expert technical recruiter evaluating screening questions. Be critical and precise.

Score the following technical screening question on three criteria (0-10 each):

## 1. Relevance (0-10): Does this question test skills actually needed for the role?
- 0-2: Completely unrelated or tests wrong domain entirely
- 3-4: Superficially mentions tech stack but tests generic programming
- 5-6: Tests general skills, only tangentially related to role focus
- 7-8: Tests relevant skills but misses the specific role requirements
- 9-10: Directly tests core skills AND tech stack listed in role description

**Red flags (subtract points):**
- Generic interview questions that could apply to any software role (-3)
- Only tests one minor technology, ignores role's main focus area (-2)
- Asks about technologies not listed in the role description (-2)

## 2. Clarity (0-10): Is the question unambiguous and well-formed?
- 0-2: Confusing, grammatically broken, impossible to answer as written
- 3-4: Multiple valid interpretations, unclear what's being asked
- 5-6: Understandable but vague about expected depth or scope
- 7-8: Clear intent but missing constraints (time, format, depth)
- 9-10: Crystal clear, unambiguous, well-scoped for a screening question

**Red flags (subtract points):**
- Multiple questions bundled into one (-2)
- "Design a system" without scope constraints (-2)
- Ambiguous technical jargon without context (-1)

## 3. Discriminative Power (0-10): Would this question distinguish good candidates from weak ones?
- 0-2: Trivial (anyone passes) OR impossible (no one passes)
- 3-4: Pure recall/trivia, doesn't test understanding
- 5-6: Tests basic knowledge, weak differentiation
- 7-8: Tests applied knowledge, good candidate separation
- 9-10: Tests trade-offs and deep understanding, strong separation

**Red flags (subtract points):**
- "Name X features of Y" style recall questions (-3)
- Yes/no or single-word answer questions (-2)
- Requires insider knowledge not in job requirements (-2)
- Too broad: would take >10 min to answer properly (-2)

Be strict. Most LLM-generated questions score 5-7, not 8-10. Reserve high scores for exceptional questions.

Return ONLY a JSON object with this exact format, no other text:
{"relevance": <0-10>, "clarity": <0-10>, "discriminative": <0-10>, "reasoning": "<brief explanation>"}
"""


def sanitize_json_string(text: str) -> str:
    """Remove or escape control characters that break JSON parsing."""
    result = []
    for char in text:
        code = ord(char)
        if code < 0x20:  # Control character range
            if code == 0x09:  # tab
                result.append(' ')
            elif code in (0x0A, 0x0D):  # newline, carriage return
                result.append(' ')
            # Skip other control characters (null, etc.)
        else:
            result.append(char)
    return ''.join(result)


def extract_json(text: str) -> str:
    """Extract JSON object from text, handling markdown and trailing content."""
    import re
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

    # Find JSON object boundaries
    start = text.find('{')
    if start == -1:
        return text

    # Find matching closing brace
    depth = 0
    in_string = False
    escape = False
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i+1]
                return sanitize_json_string(json_str)

    return sanitize_json_string(text[start:])


def judge_question(role_description: str, question: str) -> dict:
    """
    Score a question using Grok as judge (hardcoded for fair comparison).
    Returns dict with relevance, clarity, discriminative, and composite scores.
    """
    if xai_client is None:
        raise ValueError("XAI_API_KEY environment variable not set (required for judge)")

    user_prompt = f"""## Role Description
{role_description}

## Generated Screening Question
{question}

Score this question:"""

    chat = xai_client.chat.create(
        model=JUDGE_MODEL,
        messages=[system(JUDGE_SYSTEM_PROMPT)],
        response_format="json_object"
    )
    chat.append(user(user_prompt))

    response = chat.sample()
    json_str = extract_json(response.content)
    scores = JudgeResponse.model_validate_json(json_str)

    result = scores.model_dump()
    result["composite"] = (scores.relevance + scores.clarity + scores.discriminative) / 3.0
    return result


def evaluate_single_role(role_json: str, model_alias: str, checkpoint_path: Optional[str]) -> dict:
    """Evaluate a single role with a single model. Used for parallel execution."""
    role = json.loads(role_json)
    question = generate_question(role, model_alias, checkpoint_path)
    scores = judge_question(role["description"], question)
    return {
        "role_id": role["id"],
        "role_title": role["title"],
        "question": question,
        "scores": scores
    }


def evaluate_model(
    test_data_path: str,
    model_alias: str,
    checkpoint_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_workers: int = 10
) -> dict:
    """
    Evaluate a model on test data with parallel execution.

    Args:
        test_data_path: Path to test.parquet
        model_alias: Model alias (e.g., 'grok-4-1', 'claude-4-haiku', 'baseline')
        checkpoint_path: Required if model_alias is 'rl'
        num_samples: Number of samples to evaluate (None for all)
        max_workers: Number of parallel workers for API calls

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

    # Check if model is local (can't parallelize vLLM easily)
    config = MODEL_ALIASES.get(model_alias, {})
    is_local = config.get("provider") == "local"

    if is_local:
        # Sequential execution for local models
        for idx, row in df.iterrows():
            result = evaluate_single_role(row["role_json"], model_alias, checkpoint_path)
            results.append(result)
            scores = result["scores"]
            total_relevance += scores.get("relevance", 0)
            total_clarity += scores.get("clarity", 0)
            total_discriminative += scores.get("discriminative", 0)
            total_composite += scores.get("composite", 0)
            print(f"  [{model_alias}] [{idx+1}/{len(df)}] {result['role_title']}: {scores.get('composite', 0):.2f}")
    else:
        # Parallel execution for API models
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_single_role, row["role_json"], model_alias, checkpoint_path): idx
                for idx, row in df.iterrows()
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    scores = result["scores"]
                    total_relevance += scores.get("relevance", 0)
                    total_clarity += scores.get("clarity", 0)
                    total_discriminative += scores.get("discriminative", 0)
                    total_composite += scores.get("composite", 0)
                    print(f"  [{model_alias}] [{len(results)}/{len(df)}] {result['role_title']}: {scores.get('composite', 0):.2f}")
                except Exception as e:
                    print(f"  [{model_alias}] [ERROR] Role {idx}: {e}")

    n = len(results)
    return {
        "model": model_alias,
        "num_samples": n,
        "avg_relevance": total_relevance / n if n > 0 else 0,
        "avg_clarity": total_clarity / n if n > 0 else 0,
        "avg_discriminative": total_discriminative / n if n > 0 else 0,
        "avg_composite": total_composite / n if n > 0 else 0,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate question generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval.py --model grok-4-1
  python scripts/eval.py --model grok-4-1 claude-4-5-haiku gpt-5-nano
  python scripts/eval.py --model baseline rl --checkpoint checkpoints/step_100
  python scripts/eval.py --model all

Available models:
  grok-4-1       - Grok 4.1 (xAI)
  claude-4-5-haiku - Claude 4.5 Haiku (Anthropic)
  gpt-5-nano     - GPT-5 Nano (OpenAI)
  baseline       - Qwen3-4B-Instruct (local, no RL)
  rl             - RL-trained checkpoint (requires --checkpoint)
  all            - All available models
"""
    )
    parser.add_argument("--test_data", default="data/processed/test.parquet",
                        help="Path to test parquet file")
    parser.add_argument("--model", nargs="+", default=["all"],
                        help="Model(s) to evaluate (can specify multiple)")
    parser.add_argument("--checkpoint", help="Path to RL checkpoint (required for 'rl' model)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Max parallel workers for API calls (default: 10)")
    parser.add_argument("--output", default="results/eval_results.json",
                        help="Output path for results")
    parser.add_argument("--override", action="store_true",
                        help="(deprecated) Results always merge now - only specified models get updated")
    args = parser.parse_args()

    # Create results directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Expand 'all' to all available models
    models_to_eval = []
    for m in args.model:
        if m == "all":
            models_to_eval.extend(MODEL_ALIASES.keys())
        else:
            models_to_eval.append(m)

    # Remove duplicates while preserving order
    seen = set()
    models_to_eval = [m for m in models_to_eval if not (m in seen or seen.add(m))]

    # Validate models
    for m in models_to_eval:
        if m not in MODEL_ALIASES:
            raise ValueError(f"Unknown model: {m}. Available: {list(MODEL_ALIASES.keys())}")
        if m == "rl" and not args.checkpoint:
            raise ValueError("--checkpoint required for 'rl' model")

    # Separate local and API models
    local_models = [m for m in models_to_eval if MODEL_ALIASES[m]["provider"] == "local"]
    api_models = [m for m in models_to_eval if MODEL_ALIASES[m]["provider"] != "local"]

    all_results = {}

    # Run API models in parallel
    if api_models:
        print(f"\n{'='*60}")
        print(f"Evaluating API models in parallel: {', '.join(api_models)}")
        print(f"{'='*60}")

        with ThreadPoolExecutor(max_workers=len(api_models)) as executor:
            futures = {
                executor.submit(
                    evaluate_model,
                    args.test_data,
                    model,
                    args.checkpoint,
                    args.num_samples,
                    args.max_workers
                ): model
                for model in api_models
            }

            for future in as_completed(futures):
                model = futures[future]
                try:
                    all_results[model] = future.result()
                    print(f"\n[{model}] Completed!")
                except Exception as e:
                    print(f"\n[{model}] ERROR: {e}")

    # Run local models sequentially (vLLM can't easily parallelize)
    for model_alias in local_models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_alias}")
        print(f"{'='*60}")

        try:
            all_results[model_alias] = evaluate_model(
                args.test_data,
                model_alias,
                checkpoint_path=args.checkpoint,
                num_samples=args.num_samples,
                max_workers=args.max_workers
            )
        except Exception as e:
            print(f"ERROR evaluating {model_alias}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Judge model: {JUDGE_MODEL} (hardcoded for fair comparison)")

    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Samples: {result['num_samples']}")
        print(f"  Avg Relevance: {result['avg_relevance']:.2f}")
        print(f"  Avg Clarity: {result['avg_clarity']:.2f}")
        print(f"  Avg Discriminative: {result['avg_discriminative']:.2f}")
        print(f"  Avg Composite: {result['avg_composite']:.2f}")

    # Save results (always merge - only specified models get updated)
    if output_path.exists():
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        # Merge: new results override existing for same model names
        existing_results.update(all_results)
        all_results = existing_results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
