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
