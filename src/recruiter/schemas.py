"""Pydantic schemas for structured API responses."""
from pydantic import BaseModel, Field


class JudgeResponse(BaseModel):
    """Schema for LLM judge scoring response.

    Used with xai-sdk's chat.parse() to guarantee schema-compliant responses.
    """
    relevance: int = Field(ge=0, le=10, description="Relevance to role requirements (0-10)")
    clarity: int = Field(ge=0, le=10, description="Question clarity (0-10)")
    discriminative: int = Field(ge=0, le=10, description="Discriminative power (0-10)")
    reasoning: str = Field(description="Brief explanation of scores")
