"""Utilities for converting transcripts into LangChain documents."""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import get_settings


def build_documents(
    transcript_text: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> List[Document]:
    """Split a transcript into chunked LangChain documents."""
    if not transcript_text:
        return []

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " "],
    )

    metadata = metadata or {}
    docs = splitter.create_documents([transcript_text], metadatas=[metadata])
    # Ensure the video id remains on every chunk for tracing and caching.
    video_id = metadata.get("video_id")
    if video_id:
        for doc in docs:
            doc.metadata["video_id"] = video_id
    return docs
