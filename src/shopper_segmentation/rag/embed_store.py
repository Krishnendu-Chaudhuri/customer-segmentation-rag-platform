"""Embed segment cards into a local ChromaDB collection."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from shopper_segmentation.etl import OUTPUT_DIR
from shopper_segmentation.rag.build_cards import build_all_cards

CHROMA_DIR = OUTPUT_DIR / "chroma"
COLLECTION_NAME = "segment_cards"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the local sentence-transformers embedding model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def build_chroma_collection(
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Build or rebuild the ChromaDB collection from segment cards.

    Args:
        persist_dir: Directory for Chroma persistence.
        collection_name: Chroma collection name.
        model_name: Embedding model name.

    Returns:
        Populated Chroma collection.
    """
    cards = build_all_cards()
    model = get_embedding_model(model_name)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)
    documents = [str(card["content"]) for card in cards]
    ids = [f"segment_{card['segment_id']}" for card in cards]
    metadatas = [
        {
            "segment_id": int(card["segment_id"]),
            "segment_name": str(card["segment_name"]),
        }
        for card in cards
    ]
    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return collection


def get_collection(
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Return an existing Chroma collection, building it if missing.

    Args:
        persist_dir: Directory for Chroma persistence.
        collection_name: Chroma collection name.

    Returns:
        Chroma collection instance.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection(collection_name)
    except Exception:
        return build_chroma_collection(persist_dir, collection_name)


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
        Retrieved card records with content and metadata.
    """
    collection = get_collection(persist_dir, collection_name)
    model = get_embedding_model(model_name)
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    cards: list[dict[str, object]] = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        cards.append(
            {
                "segment_id": meta["segment_id"],
                "segment_name": meta["segment_name"],
                "content": doc,
                "distance": distance,
            }
        )
    return cards
