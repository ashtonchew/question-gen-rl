"""Human feedback client for online RL training."""
import time
import uuid
import logging
from typing import Tuple, Optional
import requests

logger = logging.getLogger(__name__)


class HumanFeedbackClient:
    """Client for submitting questions and receiving human feedback rewards."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: int = 300,
        poll_interval: int = 2
    ):
        """
        Initialize the human feedback client.

        Args:
            api_url: Base URL of the feedback UI API
            timeout: Max seconds to wait for human rating
            poll_interval: Seconds between polling for rating
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    def submit_and_wait(
        self,
        question: str,
        role_context: Optional[str] = None,
        role_id: Optional[str] = None
    ) -> Tuple[float, dict]:
        """
        Submit a question and wait for human rating.

        Args:
            question: The generated question to rate
            role_context: The role description context
            role_id: The role ID

        Returns:
            (reward, details_dict) where reward is 0-1
        """
        request_id = str(uuid.uuid4())

        # Submit to queue
        try:
            resp = requests.post(
                f"{self.api_url}/api/queue/submit",
                json={
                    "request_id": request_id,
                    "question": question,
                    "role_context": role_context,
                    "role_id": role_id
                },
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            if not result.get("success"):
                logger.error(f"Failed to submit question: {result.get('message')}")
                return 0.0, {"error": "submit_failed", "details": result.get("message")}
        except Exception as e:
            logger.error(f"Error submitting question: {e}")
            return 0.0, {"error": "submit_error", "details": str(e)}

        logger.info(f"Question submitted for human rating: {request_id}")

        # Poll for result
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                resp = requests.get(
                    f"{self.api_url}/api/queue/result/{request_id}",
                    timeout=10
                )
                resp.raise_for_status()
                result = resp.json()

                if result["status"] == "rated":
                    reward = result["reward"]
                    details = {
                        "relevance": result["relevance"],
                        "clarity": result["clarity"],
                        "discriminative": result["discriminative"],
                        "source": "human"
                    }
                    logger.info(f"Received human rating: {request_id}, reward={reward:.3f}")
                    return reward, details

                if result["status"] == "expired":
                    logger.warning(f"Question expired: {request_id}")
                    return 0.0, {"error": "expired"}

                # Still pending, wait and poll again
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error polling for result: {e}")
                time.sleep(self.poll_interval)

        # Timeout reached
        logger.warning(f"Timeout waiting for human rating: {request_id}")
        return 0.0, {"error": "timeout", "request_id": request_id}

    def check_health(self) -> bool:
        """Check if the feedback UI is healthy."""
        try:
            resp = requests.get(f"{self.api_url}/api/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# Global client instance (lazy initialization)
_client: Optional[HumanFeedbackClient] = None


def get_client(
    api_url: str = "http://localhost:8000",
    timeout: int = 300,
    poll_interval: int = 2
) -> HumanFeedbackClient:
    """Get or create the global human feedback client."""
    global _client
    if _client is None:
        _client = HumanFeedbackClient(api_url, timeout, poll_interval)
    return _client


def judge_question_human(
    role_description: str,
    question: str,
    api_url: str = "http://localhost:8000",
    timeout: int = 300,
    poll_interval: int = 2
) -> Tuple[float, dict]:
    """
    Get human feedback reward for a generated question.

    Drop-in replacement for judge_question() that uses human feedback.

    Args:
        role_description: The role context for evaluation
        question: The generated screening question to score
        api_url: Feedback UI API URL
        timeout: Max seconds to wait
        poll_interval: Seconds between polls

    Returns:
        (normalized_reward, details_dict) where reward is 0-1
    """
    client = get_client(api_url, timeout, poll_interval)

    # Extract role_id from description if possible
    import re
    match = re.search(r'\*\*ID:\*\*\s*(\S+)', role_description)
    role_id = match.group(1) if match else None

    return client.submit_and_wait(
        question=question,
        role_context=role_description,
        role_id=role_id
    )
