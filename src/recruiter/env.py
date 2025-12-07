"""SkyRL environment for technical question generation."""
from dataclasses import dataclass
from typing import Optional

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
        # Handle both old schema (domain) and new schema (focus, stack)
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
