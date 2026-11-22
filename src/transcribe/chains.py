"""LangChain summary chain. Summarisation is separate from the transcription request."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class Summary(BaseModel):
    title: str = Field(description="Short descriptive title for the recording")
    summary: str = Field(description="Concise summary, in the transcript's language")
    key_points: list[str] = Field(description="Main points, decisions and action items")


PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
     "You summarise transcripts of audio recordings. Use only what the transcript says. "
     "Speaker labels such as speaker_1 are anonymous; do not guess identities. "
     "Write in the same language as the transcript.")),
    ("human", "Transcript:\n\n{transcript}"),
])


def build_summary_chain(model: str) -> Runnable:
    """`prompt | ChatMistralAI.with_structured_output(Summary)`; reads MISTRAL_API_KEY."""
    from langchain_mistralai import ChatMistralAI

    return PROMPT | ChatMistralAI(model=model, temperature=0).with_structured_output(Summary)
