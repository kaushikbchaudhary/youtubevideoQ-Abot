"""RAG components for the YouTube Q&A Bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .config import Settings, get_settings


@dataclass
class VideoKnowledgeBase:
    """In-memory store of FAISS indexes per video id."""

    settings: Settings = field(default_factory=get_settings)
    _indexes: Dict[str, FAISS] = field(default_factory=dict)

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            openai_api_key=self.settings.openai_api_key,
            model=self.settings.embedding_model,
        )

    def build_index(self, video_id: str, documents: Iterable[Document]) -> FAISS:
        docs = list(documents)
        if not docs:
            raise ValueError("Cannot build an index without documents.")
        store = FAISS.from_documents(docs, self.embeddings)
        self._indexes[video_id] = store
        return store

    def get_index(self, video_id: str) -> Optional[FAISS]:
        return self._indexes.get(video_id)


def build_qa_chain(vector_store: FAISS, *, settings: Optional[Settings] = None) -> RetrievalQA:
    settings = settings or get_settings()
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.temperature,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})
    return RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")


def answer_question(
    video_id: str,
    *,
    question: str,
    knowledge_base: VideoKnowledgeBase,
    documents: Iterable[Document],
) -> str:
    """Ensure an index exists for the video and answer the question."""
    store = knowledge_base.get_index(video_id)
    if store is None:
        store = knowledge_base.build_index(video_id, documents)

    chain = build_qa_chain(store, settings=knowledge_base.settings)
    result = chain.invoke({"query": question})
    if isinstance(result, dict):
        return result.get("result") or result.get("output_text") or ""
    return str(result)
