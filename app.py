from __future__ import annotations

import streamlit as st

from src.config import Settings, get_settings
from src.qa_chain import VideoKnowledgeBase, answer_question
from src.summarizer import summarize_documents
from src.transcript_loader import build_documents
from src.youtube_client import YouTubeClient, get_metadata_and_transcript

st.set_page_config(page_title="YouTube Video Q&A Bot", page_icon="🎬", layout="wide")


def init_app_state():
    if "settings" not in st.session_state:
        try:
            st.session_state.settings = get_settings()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = VideoKnowledgeBase(settings=st.session_state.settings)
    if "videos" not in st.session_state:
        st.session_state.videos = {}
    if "qa_last_answer" not in st.session_state:
        st.session_state.qa_last_answer = None


def load_video(video_input: str):
    settings: Settings = st.session_state.settings
    video_id = YouTubeClient.parse_video_id(video_input)
    if not video_id:
        st.error("Please provide a valid YouTube URL or video ID.")
        return

    client = YouTubeClient(settings.youtube_api_key)
    with st.spinner("Fetching transcript and metadata..."):
        try:
            payload = get_metadata_and_transcript(client, video_id)
        except Exception as exc:  # noqa: BLE001 - surface precise error to user
            st.error(f"Failed to retrieve video data: {exc}")
            return

    transcript_text = payload.get("transcript")
    if not transcript_text:
        st.error("Transcript could not be fetched for this video.")
        return

    metadata = {
        "video_id": video_id,
        "title": payload.get("title"),
        "channel": payload.get("channel_title"),
        "published_at": payload.get("published_at"),
    }
    documents = build_documents(transcript_text, metadata=metadata)

    summary = None
    with st.spinner("Generating summary..."):
        summary = summarize_documents(documents)

    st.session_state.videos[video_id] = {
        "input": video_input,
        "metadata": payload,
        "documents": documents,
        "summary": summary,
    }
    st.session_state.selected_video = video_id
    st.session_state.qa_last_answer = None


@st.cache_data(show_spinner=False)
def render_metadata_card(payload: dict[str, str | None]) -> dict[str, str | None]:
    return payload


def main():
    init_app_state()

    st.title("YouTube Video Q&A Bot")
    st.caption("Summarize any public YouTube video and ask focused questions powered by LangChain + OpenAI.")

    left, right = st.columns([3, 2])
    with left:
        with st.form("video-loader", clear_on_submit=False):
            video_input = st.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            submitted = st.form_submit_button("Load video", type="primary")
        if submitted and video_input:
            load_video(video_input)

        selected_video = st.session_state.get("selected_video")
        if selected_video:
            video_state = st.session_state.videos[selected_video]
            metadata = video_state.get("metadata") or {}

            st.subheader("Summary")
            st.write(video_state.get("summary") or "No summary available.")

            st.subheader("Ask a question")
            question = st.text_input("Question", placeholder="What are the key takeaways?")
            if st.button("Get answer", type="primary"):
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    knowledge_base: VideoKnowledgeBase = st.session_state.knowledge_base
                    documents = video_state.get("documents")
                    if not documents:
                        st.error("No documents are loaded for this video. Try reloading it.")
                    else:
                        with st.spinner("Thinking..."):
                            answer = answer_question(
                                selected_video,
                                question=question,
                                knowledge_base=knowledge_base,
                                documents=documents,
                            )
                        if answer:
                            st.session_state.qa_last_answer = {
                                "question": question,
                                "answer": answer,
                            }
                        else:
                            st.warning("No answer generated. Try asking in a different way.")

            last_answer = st.session_state.get("qa_last_answer")
            if last_answer:
                st.markdown(f"**Last question:** {last_answer['question']}")
                st.write(last_answer['answer'])
    with right:
        selected_video = st.session_state.get("selected_video")
        if selected_video:
            video_state = st.session_state.videos[selected_video]
            metadata = video_state.get("metadata") or {}
            st.subheader("Video info")
            card = render_metadata_card(metadata)
            title = card.get("title") or "Unknown title"
            st.markdown(f"**{title}**")
            st.markdown(f"Channel: {card.get('channel_title') or 'Unknown'}")
            st.markdown(f"Published: {card.get('published_at') or 'Unknown'}")
            description = card.get("description")
            if description:
                with st.expander("Description", expanded=False):
                    st.write(description)
        else:
            st.info("Load a video to see its details and start asking questions.")


if __name__ == "__main__":
    main()
