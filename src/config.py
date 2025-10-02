"""Configuration helpers for the YouTube Video Q&A Bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""

    openai_api_key: str
    youtube_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    summarize_prompt_name: str = "map_reduce"
    max_summary_tokens: int = 400
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6
    temperature: float = 0.2

    @classmethod
    def load(cls, *, raise_on_missing: bool = True) -> "Settings":
        """Load settings from the environment, validating required keys."""
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        youtube_api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if raise_on_missing:
            missing = [
                name
                for name, value in (
                    ("OPENAI_API_KEY", openai_api_key),
                    ("YOUTUBE_API_KEY", youtube_api_key),
                )
                if not value
            ]
            if missing:
                joined = ", ".join(missing)
                raise RuntimeError(
                    f"Missing required environment variables: {joined}. "
                    "Populate them in a .env file or export them before running the app."
                )

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
        summarize_prompt_name = os.getenv("SUMMARIZE_PROMPT", "map_reduce").strip()
        max_summary_tokens = int(os.getenv("MAX_SUMMARY_TOKENS", "400"))
        chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        top_k = int(os.getenv("TOP_K", "6"))
        temperature = float(os.getenv("TEMPERATURE", "0.2"))

        return cls(
            openai_api_key=openai_api_key,
            youtube_api_key=youtube_api_key,
            openai_model=openai_model or "gpt-4o-mini",
            embedding_model=embedding_model or "text-embedding-3-large",
            summarize_prompt_name=summarize_prompt_name or "map_reduce",
            max_summary_tokens=max_summary_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            temperature=temperature,
        )


_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Return cached settings to avoid repeated env lookups."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
    return _SETTINGS
