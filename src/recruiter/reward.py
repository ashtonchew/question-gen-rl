"""Reward function using Grok API as judge."""
import os
import json
import httpx
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

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
