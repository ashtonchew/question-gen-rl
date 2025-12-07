"""Prompt formatting for question generation.

Centralizes all prompt logic - change prompts here without regenerating data.
"""
import json
from typing import Union, List, Dict

# System message for the question generator
SYSTEM_MESSAGE = """You are an expert technical recruiter creating screening questions.
Your goal is to generate clear, relevant questions that test practical knowledge.
Questions should be answerable in 2-5 minutes and appropriate for the candidate's level."""


def format_prompt(role: Union[dict, str]) -> List[Dict[str, str]]:
    """Format role data into chat conversation for instruction-tuned models.

    Args:
        role: Role dict or JSON string containing role data

    Returns:
        List of message dicts with system/user roles for chat models
    """
    if isinstance(role, str):
        role = json.loads(role)

    # Handle both old schema (domain) and new schema (focus, stack)
    focus = role.get('focus', role.get('domain', 'backend'))
    stack = role.get('stack', [])
    stack_str = ', '.join(stack) if stack else 'Not specified'
    key_skills = role.get('key_skills', [])
    skills_str = ', '.join(key_skills) if key_skills else 'Not specified'

    user_message = f"""Generate ONE technical screening question for this role:

## Role: {role['title']}
**ID:** {role['id']}
**Level:** {role['level']}
**Focus Area:** {focus}
**Tech Stack:** {stack_str}

**Description:** {role['description']}

**Key Skills:** {skills_str}

Question:"""

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]
