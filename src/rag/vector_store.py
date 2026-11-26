"""Optional ChromaDB + LangChain RAG indexer for full vector-store workflows.

Requires optional packages listed in `requirements-rag.txt`.
Core ML prediction and local grounded retrieval work without running this script.
"""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT
from src.rag.assistant import build_knowledge_chunks

CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def build_chroma_index(persist_dir: Path = CHROMA_DIR) -> int:
    """Chunk and index project artifacts into a local persistent ChromaDB collection."""
    try:
        import chromadb
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "Optional RAG dependencies are not installed. Install them via: "
            "`pip install -r requirements-rag.txt`"
        ) from exc

    chunks = build_knowledge_chunks()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    docs, metadatas, ids = [], [], []
    counter = 0
    for ch in chunks:
        sub_chunks = splitter.split_text(f"{ch.title}\n\n{ch.content}")
        for part_idx, text in enumerate(sub_chunks):
            docs.append(text)
            metadatas.append({"source": ch.source, "title": ch.title, "chunk_id": ch.chunk_id})
            ids.append(f"{ch.chunk_id}_{part_idx}_{counter}")
            counter += 1

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="telco_churn_knowledge")
    collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
    return len(docs)


if __name__ == "__main__":
    n = build_chroma_index()
    print(f"Indexed {n} grounded knowledge chunks into ChromaDB at {CHROMA_DIR}")
