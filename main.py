from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/lao_hac.txt",
    "data/chi_pheo.txt",
    "data/doi_mat.txt",
    "data/doi_thua.txt",
    "data/mot_bua_no.txt",
    "data/tre_con_khong_duoc_an_thit_cho.txt",
]

STORY_METADATA = {
    "lao_hac": {"title": "Lão Hạc", "characters": ["Lão Hạc", "Cậu Vàng", "Ông Giáo", "Binh Tư"], "tags": ["bán chó"]},
    "chi_pheo": {"title": "Chí Phèo", "characters": ["Chí Phèo", "Thị Nở", "Bá Kiến", "Lý Cường"], "tags": ["tha hóa", "làng Vũ Đại", "bát cháo hành"]},
    "doi_mat": {"title": "Đôi Mắt", "characters": ["Hoàng", "Độ"], "tags": ["trí thức", "kháng chiến"]},
    "doi_thua": {"title": "Đời Thừa", "characters": ["Hộ", "Từ"], "tags": ["văn chương", "bi kịch trí thức"]},
    "mot_bua_no": {"title": "Một Bữa No", "characters": ["Bà lão", "Bà phó Thụ"], "tags": []},
    "tre_con_khong_duoc_an_thit_cho": {"title": "Trẻ con không được ăn thịt chó", "characters": [], "tags": []}
}

# --- CÁC CÂU HỎI BENCHMARK THEO BÁO CÁO ---
BENCHMARK_QUERIES = [
    {"query": "Chí Phèo chửi ai?", "gold": "Chí Phèo chửi trời, chửi đời, chửi cả làng Vũ Đại"},
    {"query": "Thị Nở nấu gì cho Chí Phèo?", "gold": "Thị Nở nấu cháo hành"},
    {"query": "Chí Phèo ăn vạ ai?", "gold": "Ăn vạ đội Tảo và bá Kiến"},
    {"query": "Ai bắt cậu Vàng?", "gold": "Lão Hạc bán cậu Vàng"},
    {"query": "Bi kịch của Hộ trong Đời Thừa là gì?", "gold": "Nhà văn mất lý tưởng vì gánh nặng cơm áo"}
]

# THAY ĐỔI: Dùng số lượng câu thay vì số lượng ký tự
max_sentences = 3

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions or not path.exists():
            continue

        # THAY ĐỔI: Import và sử dụng SentenceChunker
        from src.chunking import SentenceChunker
        content = path.read_text(encoding="utf-8")
        
        chunks = SentenceChunker(max_sentences_per_chunk=max_sentences).chunk(content)
        
        file_key = path.stem 
        meta = STORY_METADATA.get(file_key, {})
        
        for i, text_chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{file_key}_part_{i}",
                    content=text_chunk,
                    metadata={
                        "source": str(path),
                        "title": meta.get("title", file_key),
                        "characters": ", ".join(meta.get("characters", [])),
                        "tags": ", ".join(meta.get("tags", [])),
                        "chunk_idx": i
                    },
                )
            )
    return documents

def demo_llm(prompt: str) -> str:
    import os
    try:
        from openai import OpenAI
    except ImportError:
        return "[LỖI] Xin hãy chạy lệnh: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-openai-key-here":
        return "[LỖI] Bạn chưa dán key vào biến OPENAI_API_KEY trong file .env kìa!"
    
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý ảo. Nhiệm vụ của bạn là đọc kỹ đoạn [Context] được cấp và trả lời câu hỏi thật ngắn gọn, đi thẳng vào trọng tâm (chỉ 1-2 câu). Trả lời bằng Tiếng Việt. Nếu ngữ cảnh không nhắc đến, hãy nói 'Không đủ thông tin'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LỖI API] {str(e)}"

def run_benchmark(sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    
    import pickle
    db_file = "vector_db.pkl"
    if os.path.exists(db_file):
        print(f"\n[🚀 CACHE] Tìm thấy vector_db.pkl. Đang nạp lại từ ổ cứng...")
        with open(db_file, "rb") as f:
            store._store = pickle.load(f)
    else:
        # THAY ĐỔI: Cập nhật câu log cho chính xác
        print(f"\n[⏳ CACHE] Chưa có file cache. Đang tạo Embedding từ đầu với SentenceChunker (max {max_sentences} câu/chunk)...")
        docs = load_documents_from_files(files)
        if not docs:
            print("\nKhông tìm thấy file TXT nào trong data/.")
            return 1
            
        store.add_documents(docs)
        with open(db_file, "wb") as f:
            pickle.dump(store._store, f)
        print(f"[🔥 CACHE] Đã lưu thành công {store.get_collection_size()} chunks vào {db_file}.")

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    # --- KHỞI TẠO HỆ THỐNG LOGGING ---
    current_time = datetime.datetime.now()
    # Tạo thư mục logs nếu chưa có
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/benchmark_proof.txt"  
    
    with open(log_filename, "w", encoding="utf-8") as log_file:
        def lprint(text=""):
            print(text)
            log_file.write(text + "\n")

        lprint("="*75)
        lprint(" 🚀 BÁO CÁO KẾT QUẢ CHẠY BENCHMARK RAG TỰ ĐỘNG (LAB 7)")
        lprint(f" Sinh viên thực hiện : Nguyễn Hoàng Việt Hùng - 2A202600164 - Nhóm 08")
        lprint(f" Chiến lược Chunking : SentenceChunker ({max_sentences} câu/chunk)")
        lprint(f" Vector Store Backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
        lprint("="*75 + "\n")

        for i, test in enumerate(BENCHMARK_QUERIES, start=1):
            query = test["query"]
            lprint(f"[{i}/5] QUERY: {query}")
            
            search_results = store.search(query, top_k=15)
            if search_results:
                top1_score = search_results[0]['score']
                top1_chunk_preview = search_results[0]['content'][:80].replace('\n', ' ') + "..." 
            else:
                top1_score = 0.0
                top1_chunk_preview = "Không tìm thấy"

            agent_answer = agent.answer(query, top_k=15)

            lprint(f"  ├─ Top-1 Score : {top1_score:.3f}")
            lprint(f"  ├─ Top-1 Chunk : \"{top1_chunk_preview}\"")
            lprint(f"  ├─ Gold Answer : {test['gold']}")
            lprint(f"  └─ Agent Ans   : {agent_answer}")
            lprint("-" * 75)

        lprint(f"\n✅ HOÀN TẤT BENCHMARK! (Log được lưu tại: {log_filename})")

    return 0

def main() -> int:
    return run_benchmark()

if __name__ == "__main__":
    raise SystemExit(main())