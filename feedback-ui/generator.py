"""Question generation using Grok API for the feedback UI."""
import os
import json
import random
import logging
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system, user

# Load .env from repo root (parent directory)
_repo_root = Path(__file__).parent.parent
load_dotenv(_repo_root / ".env")

logger = logging.getLogger(__name__)

# Grok client setup (same pattern as src/recruiter/reward.py)
_api_key = os.environ.get("XAI_API_KEY")
if not _api_key:
    logger.warning("XAI_API_KEY not set - question generation will fail")

client = Client(api_key=_api_key, timeout=600) if _api_key else None

# Role data cache
_roles: List[Dict] = []

# Pre-warming cache for questions
_question_cache: Queue[Tuple[Dict, str]] = Queue(maxsize=3)
_cache_lock = threading.Lock()
_warming_thread: Optional[threading.Thread] = None
_stop_warming = threading.Event()

# System prompt for question generation (same as src/recruiter/prompts.py)
GENERATOR_SYSTEM_PROMPT = """You are an expert technical recruiter creating screening questions.
Your goal is to generate clear, relevant questions that test practical knowledge.
Questions should be answerable in 2-5 minutes and appropriate for the candidate's level."""


def load_roles(roles_path: str = None) -> List[Dict]:
    """Load roles from backend_roles.json (cached in memory)."""
    global _roles
    if _roles:
        return _roles

    if roles_path is None:
        # Look in parent directory's data folder
        roles_path = Path(__file__).parent.parent / "data" / "backend_roles.json"

    with open(roles_path) as f:
        _roles = json.load(f)

    logger.info(f"Loaded {len(_roles)} roles from {roles_path}")
    return _roles


def get_random_role() -> Dict:
    """Get a random role from the cached roles."""
    if not _roles:
        load_roles()
    return random.choice(_roles)


def format_role_prompt(role: Dict) -> str:
    """Format role into user prompt (same pattern as src/recruiter/prompts.py)."""
    focus = role.get('focus', role.get('domain', 'backend'))
    stack = role.get('stack', [])
    stack_str = ', '.join(stack) if stack else 'Not specified'
    key_skills = role.get('key_skills', [])
    skills_str = ', '.join(key_skills) if key_skills else 'Not specified'

    return f"""Generate ONE technical screening question for this role:

## Role: {role['title']}
**ID:** {role['id']}
**Level:** {role['level']}
**Focus Area:** {focus}
**Tech Stack:** {stack_str}

**Description:** {role['description']}

**Key Skills:** {skills_str}

Question:"""


def generate_question(role: Dict) -> str:
    """Generate a question for the given role using Grok API.

    Args:
        role: Role dictionary from backend_roles.json

    Returns:
        Generated question string, or error message if generation fails
    """
    if client is None:
        return "[Error: XAI_API_KEY not set - cannot generate questions]"

    user_prompt = format_role_prompt(role)

    try:
        chat = client.chat.create(
            model="grok-4-1-fast-reasoning",
            messages=[system(GENERATOR_SYSTEM_PROMPT)]
        )
        chat.append(user(user_prompt))
        response = chat.sample()
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        return f"[Error generating question: {str(e)}]"


# ============== Pre-warming Cache System ==============

def _warm_cache():
    """Background thread to keep the question cache full."""
    logger.info("Question pre-warming thread started")
    while not _stop_warming.is_set():
        try:
            # Only generate if cache not full
            if _question_cache.qsize() < 3:
                role = get_random_role()
                question = generate_question(role)
                if not question.startswith("[Error"):
                    _question_cache.put((role, question), timeout=1)
                    logger.info(f"Pre-warmed question for {role['id']} (cache size: {_question_cache.qsize()})")
            else:
                # Cache is full, wait a bit
                _stop_warming.wait(0.5)
        except Exception as e:
            logger.error(f"Error in pre-warming thread: {e}")
            _stop_warming.wait(1)
    logger.info("Question pre-warming thread stopped")


def start_warming():
    """Start the background pre-warming thread."""
    global _warming_thread
    with _cache_lock:
        if _warming_thread is None or not _warming_thread.is_alive():
            _stop_warming.clear()
            _warming_thread = threading.Thread(target=_warm_cache, daemon=True)
            _warming_thread.start()
            logger.info("Started question pre-warming")


def stop_warming():
    """Stop the background pre-warming thread."""
    _stop_warming.set()
    if _warming_thread:
        _warming_thread.join(timeout=2)


def get_prewarmed_question() -> Tuple[Dict, str]:
    """Get a pre-warmed question from cache, or generate one if cache is empty.

    Returns:
        (role, question) tuple
    """
    # Ensure warming is running
    start_warming()

    try:
        # Try to get from cache (non-blocking)
        role, question = _question_cache.get_nowait()
        logger.info(f"Served pre-warmed question for {role['id']} (cache size: {_question_cache.qsize()})")
        return role, question
    except Empty:
        # Cache empty, generate synchronously (fallback)
        logger.warning("Cache empty, generating synchronously")
        role = get_random_role()
        question = generate_question(role)
        return role, question


def get_cache_status() -> dict:
    """Get the current cache status."""
    return {
        "size": _question_cache.qsize(),
        "max_size": 3,
        "warming_active": _warming_thread is not None and _warming_thread.is_alive()
    }
