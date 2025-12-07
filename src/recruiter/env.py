"""SkyRL environment for technical question generation."""
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
        self._prompt: Optional[str] = None
        self._role_id: Optional[str] = None

    def _extract_prompt_text(self, prompt: List[Dict[str, Any]]) -> str:
        """Extract text content from conversation format prompt."""
        # SkyRL passes prompt as ConversationType (List of message dicts)
        # Extract the user message content
        for msg in prompt:
            if msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: if no user message, try first message or join all
        if prompt:
            return prompt[0].get("content", str(prompt))
        return ""

    def _extract_role_id(self, prompt_text: str) -> Optional[str]:
        """Extract role_id from prompt text (format: **ID:** <id>)."""
        match = re.search(r'\*\*ID:\*\*\s*(\S+)', prompt_text)
        return match.group(1) if match else None

    def init(self, prompt):
        """Initialize the environment with the prompt from the dataset.

        Args:
            prompt: The prompt/conversation from the dataset (ConversationType)

        Returns:
            Tuple of (prompt, metadata_dict) as expected by SkyRL
        """
        # Store the prompt text for use in step()
        self._prompt = self._extract_prompt_text(prompt)
        self._role_id = self._extract_role_id(self._prompt)
        return prompt, {}

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

        # Guard against step() being called before init() or with empty prompt
        if not self._prompt:
            return BaseTextEnvStepOutput(
                observation="",
                reward=-1.0,
                terminated=True,
                truncated=False,
                info={"error": "step() called before init()"}
            )

        # Get reward from judge - use stored prompt as role context
        reward, judge_details = judge_question(
            role_description=self._prompt,
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
                "role_id": self._role_id,
            }
        )
