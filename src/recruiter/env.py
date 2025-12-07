"""SkyRL environment for technical question generation."""
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .reward import judge_question
from .human_feedback import judge_question_human


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

    def _extract_prompt_text(self, prompt) -> str:
        """Extract text content from prompt (handles both str and conversation format)."""
        # Handle plain string prompt
        if isinstance(prompt, str):
            return prompt

        # Handle conversation format (List of message dicts)
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            # Fallback: first message content or string representation
            if prompt and isinstance(prompt[0], dict):
                return prompt[0].get("content", str(prompt))

        return str(prompt)

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
                observations=[],  # Empty list for single-turn
                reward=-0.5,      # Penalty for too-short output
                done=True,
                metadata={"error": "Question too short"}
            )

        if len(question) > self.config.max_question_length:
            question = question[:self.config.max_question_length]

        # Guard against missing or empty prompt
        if not self._prompt:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=-1.0,
                done=True,
                metadata={"error": "No prompt available (init() not called or prompt extraction failed)"}
            )

        # Get reward from judge - use stored prompt as role context
        reward, judge_details = judge_question(
            role_description=self._prompt,
            question=question
        )

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,  # Single-turn: always done
            metadata={
                "question": question,
                "judge_scores": judge_details,
                "role_id": self._role_id,
            }
        )


@dataclass
class QuestionGenOnlineEnvConfig:
    """Configuration for the online question generation environment."""
    max_question_length: int = 500
    min_question_length: int = 20
    api_url: str = "http://localhost:8000"
    timeout: int = 300
    poll_interval: int = 2


class QuestionGenOnlineEnv(BaseTextEnv):
    """
    Online environment for training with human feedback.

    Single-turn environment:
    - State: Role description prompt
    - Action: Model generates a screening question
    - Reward: Human feedback score (0-1), waits for rating
    - Done: Always True after one generation
    """

    def __init__(self, config: Optional[QuestionGenOnlineEnvConfig] = None):
        super().__init__()
        self.config = config or QuestionGenOnlineEnvConfig()
        self._prompt: Optional[str] = None
        self._role_id: Optional[str] = None

    def _extract_prompt_text(self, prompt) -> str:
        """Extract text content from prompt (handles both str and conversation format)."""
        if isinstance(prompt, str):
            return prompt

        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            if prompt and isinstance(prompt[0], dict):
                return prompt[0].get("content", str(prompt))

        return str(prompt)

    def _extract_role_id(self, prompt_text: str) -> Optional[str]:
        """Extract role_id from prompt text (format: **ID:** <id>)."""
        match = re.search(r'\*\*ID:\*\*\s*(\S+)', prompt_text)
        return match.group(1) if match else None

    def init(self, prompt):
        """Initialize the environment with the prompt from the dataset."""
        self._prompt = self._extract_prompt_text(prompt)
        self._role_id = self._extract_role_id(self._prompt)
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process the model's generated question and get human feedback.

        Args:
            action: The generated question from the model

        Returns:
            BaseTextEnvStepOutput with reward from human and done=True
        """
        question = action.strip()

        # Basic validation
        if len(question) < self.config.min_question_length:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=-0.5,
                done=True,
                metadata={"error": "Question too short"}
            )

        if len(question) > self.config.max_question_length:
            question = question[:self.config.max_question_length]

        if not self._prompt:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=-1.0,
                done=True,
                metadata={"error": "No prompt available"}
            )

        # Get reward from human feedback (blocks until rated)
        reward, judge_details = judge_question_human(
            role_description=self._prompt,
            question=question,
            api_url=self.config.api_url,
            timeout=self.config.timeout,
            poll_interval=self.config.poll_interval
        )

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={
                "question": question,
                "judge_scores": judge_details,
                "role_id": self._role_id,
                "source": "human"
            }
        )
