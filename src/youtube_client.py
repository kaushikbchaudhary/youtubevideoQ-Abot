"""Helpers for interacting with the YouTube Data API and transcripts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterable, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import (  # type: ignore
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERN = re.compile(r"(?:(?:v=|\/)([0-9A-Za-z_-]{11}))")


@dataclass
class VideoMetadata:
    """Subset of metadata about a YouTube video we care about."""

    video_id: str
    title: str
    channel_title: str
    description: str
    published_at: str
    thumbnail_url: Optional[str]


class YouTubeClient:
    """Thin wrapper around the YouTube Data API for metadata lookups."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("A valid YouTube API key is required.")
        self._api_key = api_key

    @cached_property
    def _service(self):  # pragma: no cover - network side effect hidden in runtime
        return build("youtube", "v3", developerKey=self._api_key)

    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Fetch basic metadata for a video via the YouTube Data API."""
        try:
            response = (
                self._service.videos()
                .list(part="snippet", id=video_id, maxResults=1)
                .execute()
            )
        except HttpError as exc:  # pragma: no cover - requires API call
            logger.error("YouTube API error: %s", exc)
            raise

        items = response.get("items", []) if response else []
        if not items:
            return None

        snippet = items[0]["snippet"]
        thumbnails = snippet.get("thumbnails", {}) if snippet else {}
        high_thumb = thumbnails.get("high") or thumbnails.get("medium")
        return VideoMetadata(
            video_id=video_id,
            title=snippet.get("title", ""),
            channel_title=snippet.get("channelTitle", ""),
            description=snippet.get("description", ""),
            published_at=snippet.get("publishedAt", ""),
            thumbnail_url=(high_thumb or thumbnails.get("default") or {}).get("url"),
        )

    @staticmethod
    def parse_video_id(url_or_id: str) -> Optional[str]:
        """Extract the 11-character YouTube video id from a URL or raw id."""
        if not url_or_id:
            return None
        candidate = url_or_id.strip()
        if len(candidate) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
            return candidate
        match = _VIDEO_ID_PATTERN.search(candidate)
        return match.group(1) if match else None

    @staticmethod
    def fetch_transcript(
        video_id: str,
        languages: Optional[Iterable[str]] = None,
        *,
        chunk_separator: str = " ",
    ) -> str:
        """Fetch the transcript for a video, returning a merged string."""
        if not languages:
            languages = ("en", "en-US", "en-GB")
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=list(languages)
            )
        except TranscriptsDisabled as exc:
            raise RuntimeError("Transcripts are disabled for this video.") from exc
        except NoTranscriptFound as exc:
            raise RuntimeError("No transcript could be found for this video.") from exc

        parts: List[str] = []
        for chunk in transcript_list:
            text = chunk.get("text")
            if text:
                parts.append(text.replace("\n", " ").strip())
        return chunk_separator.join(parts)


def get_metadata_and_transcript(
    client: YouTubeClient,
    video_id: str,
    *,
    languages: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[str]]:
    """Convenience helper to fetch metadata and transcript in one call."""
    metadata = client.get_video_metadata(video_id)
    transcript_text = YouTubeClient.fetch_transcript(video_id, languages=languages)
    payload: Dict[str, Optional[str]] = {
        "transcript": transcript_text,
    }
    if metadata:
        payload.update(
            {
                "title": metadata.title,
                "channel_title": metadata.channel_title,
                "description": metadata.description,
                "published_at": metadata.published_at,
                "thumbnail_url": metadata.thumbnail_url,
            }
        )
    return payload
