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

# Initialize xai-sdk client with 10 minute timeout
# SDK has built-in retry with exponential backoff
_api_key = os.environ.get("XAI_API_KEY")
if not _api_key:
    logger.warning("XAI_API_KEY not set - judge_question will fail")

client = Client(api_key=_api_key, timeout=600) if _api_key else None

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
"""


def judge_question(role_description: str, question: str) -> Tuple[float, dict]:
    """
    Call Grok API to score a generated question.

    Uses xai-sdk which has built-in retry with exponential backoff.

    Args:
        role_description: The role context for evaluation
        question: The generated screening question to score

    Returns:
        (normalized_reward, details_dict) where reward is 0-1
    """
    if client is None:
        logger.error("XAI_API_KEY not set, cannot call judge API")
        return 0.0, {"error": "XAI_API_KEY not set"}

    user_prompt = f"""## Role Description
{role_description}

## Generated Screening Question
{question}

Score this question:"""

    try:
        chat = client.chat.create(
            model="grok-4-1-fast-non-reasoning",
            messages=[system(JUDGE_SYSTEM_PROMPT)]
        )
        chat.append(user(user_prompt))

        # Use structured outputs - guaranteed schema compliance
        response, scores = chat.parse(JudgeResponse)

        # Normalize to 0-1 range, weighted average
        reward = (scores.relevance + scores.clarity + scores.discriminative) / 30.0

        return reward, scores.model_dump()

    except Exception as e:
        logger.error(f"Error calling judge API: {e}")
        return 0.0, {"error": str(type(e).__name__), "details": str(e)}
