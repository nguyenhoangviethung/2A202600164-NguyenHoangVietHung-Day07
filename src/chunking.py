# src/chunking.pyfrom __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Tách câu dựa trên các dấu kết thúc câu (. ! ? hoặc .\n)
        sentences = re.split(r'(?<=[.!?]) +|\.\n', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_text = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk_text)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Nếu đã đủ ngắn, trả về luôn
        if len(current_text) <= self.chunk_size:
            return [current_text]
            
        # Nếu hết separator để cắt -> Dùng Fixed size fallback
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        
        # Nếu separator là rỗng -> Cắt theo ký tự
        if sep == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        splits = current_text.split(sep)
        chunks = []
        current_chunk = splits[0]

        for s in splits[1:]:
            if len(current_chunk) + len(sep) + len(s) <= self.chunk_size:
                current_chunk += sep + s
            else:
                chunks.append(current_chunk)
                current_chunk = s
        chunks.append(current_chunk)

        final_chunks = []
        # Tiếp tục đệ quy cho các chunk vẫn còn lố size
        for c in chunks:
            if len(c) > self.chunk_size:
                final_chunks.extend(self._split(c, remaining_separators[1:]))
            else:
                final_chunks.append(c)
                
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fs_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        sc_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        rc_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def get_stats(chunks_list: list[str]) -> dict:
            return {
                "count": len(chunks_list),
                "avg_length": sum(len(c) for c in chunks_list) / len(chunks_list) if chunks_list else 0,
                "chunks": chunks_list
            }

        return {
            "fixed_size": get_stats(fs_chunks),
            "by_sentences": get_stats(sc_chunks),
            "recursive": get_stats(rc_chunks)
        }