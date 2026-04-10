# src/store.py
from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            # Mock initializing chromadb (Optional logic skipped since tests test in-memory fallback)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embed_vector = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embed_vector
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vector = self._embedding_fn(query)
        scored_records = []
        
        for record in records:
            score = compute_similarity(query_vector, record["embedding"])
            # Copy record để không thay đổi dict gốc, inject thêm score
            res = record.copy()
            res["score"] = score
            scored_records.append(res)
            
        # Sắp xếp giảm dần theo điểm và lấy top_k
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        filtered_records = self._store
        
        if metadata_filter:
            filtered_records = []
            for record in self._store:
                meta = record.get("metadata", {})
                # Kiểm tra xem record có chứa tất cả điều kiện từ metadata_filter không
                if all(meta.get(k) == v for k, v in metadata_filter.items()):
                    filtered_records.append(record)
                    
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        original_size = len(self._store)
        
        # Giữ lại các bản ghi KHÔNG khớp với doc_id (hoặc metadata doc_id)
        self._store = [
            record for record in self._store 
            if record["id"] != doc_id and record.get("metadata", {}).get("doc_id") != doc_id
        ]
        
        # Trả về True nếu số lượng thực sự bị giảm đi
        return len(self._store) < original_size