# YouTube Video Q&A Bot

Interactive Streamlit app that retrieves transcripts for public YouTube videos, summarizes the content, and answers custom questions using a LangChain Retrieval-Augmented Generation (RAG) pipeline powered by OpenAI models.

## Features
- Fetch public video transcripts and metadata via the YouTube Data API and `youtube-transcript-api`.
- Automatic text chunking and vector indexing with FAISS for rapid semantic search.
- One-click transcript summarization with OpenAI chat models.
- Conversational Q&A grounded in the transcript using LangChain retrieval chains.
- Streamlit UI for loading videos, viewing summaries, and asking follow-up questions.

## Prerequisites
- Python 3.9 or later.
- API keys for:
  - [OpenAI](https://platform.openai.com/) — used for both chat completions and embeddings.
  - [YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started).

## Quickstart
1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment variables**
   - Duplicate the provided `.env` file and replace placeholder values with your own keys, or export them directly in your shell.
3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
4. **Use the app**
   - Paste a YouTube URL or video ID, click *Load video*, review the generated summary, and ask questions about the content.

## Project Structure
```
.
├── app.py                  # Streamlit UI entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variable placeholders
└── src/
    ├── __init__.py
    ├── config.py           # Settings loader (env variables + defaults)
    ├── youtube_client.py   # YouTube Data API + transcript helpers
    ├── transcript_loader.py# Transcript chunking into LangChain documents
    ├── summarizer.py       # LangChain summarization chain
    └── qa_chain.py         # RAG pipeline for Q&A over transcript chunks
```

## Notes & Tips
- The app only works with videos that have accessible transcripts (auto-generated or uploaded). Private or age-restricted videos may fail to provide transcripts.
- For larger transcripts you can tune chunk sizes, overlap, and retrieval depth via the optional values in `.env`.
- Consider enabling Streamlit caching or persistent storage if you plan to reuse embeddings across sessions.

## Future Enhancements
- Multi-language transcript support with automatic language detection.
- Persisting FAISS indexes for offline reuse.
- Support for batching multiple videos into a single knowledge base.
# youtubevideoQ-Abot
