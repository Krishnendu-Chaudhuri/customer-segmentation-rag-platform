"""LangChain-backed vector store for segment card retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from shopper_segmentation.etl import OUTPUT_DIR
from shopper_segmentation.rag.build_cards import build_all_cards

CHROMA_DIR = OUTPUT_DIR / "chroma"
COLLECTION_NAME = "segment_cards"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return the configured HuggingFace embedding model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        LangChain HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def _score_to_distance(score: float) -> float:
    """Convert a LangChain similarity score to a distance-like metric.

    LangChain returns higher scores for more similar documents. The legacy
    embed_store API exposed Chroma distances where lower values mean closer
    matches. When the score appears normalized to [0, 1], invert it.

    Args:
        score: Similarity score from similarity_search_with_score.

    Returns:
        Distance value suitable for the retrieve_cards response contract.
    """
    if 0.0 <= score <= 1.0:
        return 1.0 - score
    return float(score)


def build_vectorstore(
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> Chroma:
    """Build or rebuild the persisted Chroma vector store from segment cards.

    Args:
        persist_dir: Directory for Chroma persistence.
        collection_name: Chroma collection name.
        model_name: Embedding model name.

    Returns:
        Populated LangChain Chroma vector store.
    """
    cards = build_all_cards()
    persist_dir.mkdir(parents=True, exist_ok=True)

    texts = [str(card["content"]) for card in cards]
    ids = [f"segment_{card['segment_id']}" for card in cards]
    metadatas = [
        {
            "segment_id": int(card["segment_id"]),
            "segment_name": str(card["segment_name"]),
        }
        for card in cards
    ]

    return Chroma.from_texts(
        texts=texts,
        embedding=_get_embeddings(model_name),
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )


def get_vectorstore(
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> Chroma:
    """Return an existing vector store, building it if missing.

    Args:
        persist_dir: Directory for Chroma persistence.
        collection_name: Chroma collection name.
        model_name: Embedding model name.

    Returns:
        LangChain Chroma vector store instance.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(model_name),
        persist_directory=str(persist_dir),
    )
    if store._collection.count() == 0:
        return build_vectorstore(persist_dir, collection_name, model_name)
    return store


def retrieve_cards(
    query: str,
    top_k: int = 3,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> list[dict[str, object]]:
    """Retrieve top-k segment cards for a user query.

    Args:
        query: Natural language question.
        top_k: Number of cards to retrieve.
        persist_dir: Chroma persistence directory.
        collection_name: Chroma collection name.
        model_name: Embedding model name.

    Returns:
        Retrieved card records with segment_id, segment_name, content, distance.
    """
    store = get_vectorstore(persist_dir, collection_name, model_name)
    results = store.similarity_search_with_score(query, k=top_k)

    cards: list[dict[str, object]] = []
    for document, score in results:
        metadata = document.metadata or {}
        cards.append(
            {
                "segment_id": metadata.get("segment_id"),
                "segment_name": metadata.get("segment_name"),
                "content": document.page_content,
                "distance": _score_to_distance(float(score)),
            }
        )
    return cards
