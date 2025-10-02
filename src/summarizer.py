"""Summarization pipeline built on LangChain."""

from __future__ import annotations

from typing import Iterable

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import get_settings


def build_summarizer():
    """Construct the LangChain summarize chain."""
    settings = get_settings()
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.temperature,
        max_tokens=settings.max_summary_tokens,
    )
    return load_summarize_chain(llm, chain_type=settings.summarize_prompt_name)


def summarize_documents(documents: Iterable[Document]) -> str:
    """Produce a concise summary for the provided documents."""
    docs = list(documents)
    if not docs:
        return "No transcript available to summarize."

    chain = build_summarizer()
    result = chain.invoke({"input_documents": docs})
    if isinstance(result, dict) and "output_text" in result:
        return result["output_text"].strip()
    if isinstance(result, str):
        return result.strip()
    return "Unable to generate summary."
