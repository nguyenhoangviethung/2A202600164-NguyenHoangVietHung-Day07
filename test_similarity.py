import os
import datetime
from sentence_transformers import SentenceTransformer

# Import hàm tính cosine similarity bạn đã code trong src.chunking
try:
    from src.chunking import compute_similarity
except ImportError:
    print("❌ Không tìm thấy src.chunking. Hãy chắc chắn bạn chạy code ở thư mục gốc.")
    exit(1)

LOCAL_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"  # Mô hình local embedder (Sentence Transformers)
# Danh sách 5 cặp câu từ Báo Cáo
PAIRS = [
    ("Trời mưa rất to", "Mưa rơi nặng hạt", "high"),
    ("Mùa hè nóng nực", "Mùa đông lạnh giá", "low"),
    ("Tôi rất yêu bóng đá", "Bóng đá là môn tôi thích nhất", "high"),
    ("Trái đất quay quanh mặt trời", "Gà là động vật đẻ trứng", "low"),
    ("Tôi ghét ăn cá", "Tôi cực kì thích ăn cá", "low (surprise)"),
]

def run_similarity_test():
    print("⏳ Đang tải mô hình {LOCAL_EMBEDDING_MODEL}...")
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    # Khởi tạo file log
    current_time = datetime.datetime.now()
    log_filename = f"logs/similarity_proof.txt"

    with open(log_filename, "w", encoding="utf-8") as log_file:
        def lprint(text=""):
            print(text)
            log_file.write(text + "\n")

        lprint("="*75)
        lprint(" 🧪 BÁO CÁO TEST: COSINE SIMILARITY PREDICTIONS (LAB 7)")
        lprint(f" Sinh viên thực hiện : Nguyễn Hoàng Việt Hùng - 2A202600164 - Nhóm 08")
        lprint(f" Mô hình sử dụng     : {LOCAL_EMBEDDING_MODEL}")
        lprint("="*75 + "\n")

        for i, (sent_a, sent_b, prediction) in enumerate(PAIRS, start=1):
            vec_a = model.encode(sent_a).tolist()
            vec_b = model.encode(sent_b).tolist()

            score = compute_similarity(vec_a, vec_b)

            lprint(f"[{i}/5] Dự đoán: {prediction.upper()}")
            lprint(f"  ├─ Câu A : \"{sent_a}\"")
            lprint(f"  ├─ Câu B : \"{sent_b}\"")
            lprint(f"  └─ Score : {score:.4f}")
            lprint("-" * 75)
        lprint(f"\n✅ HOÀN TẤT! (Log được lưu tại: {log_filename})")

if __name__ == "__main__":
    run_similarity_test()