import os
import datetime
from pathlib import Path

try:
    from src.chunking import ChunkingStrategyComparator
except ImportError:
    print("❌ Lỗi Import: Không tìm thấy module 'src.chunking'. Hãy đảm bảo bạn đang chạy script ở thư mục gốc của project.")
    exit(1)

def run_baseline(chunk_size=500):
    file_path = Path("data/chi_pheo.txt")
    
    if not file_path.exists():
        print(f"❌ Không tìm thấy file: {file_path}")
        print("💡 Hãy đảm bảo bạn đã tạo thư mục 'data/' và đặt file 'chi_pheo.txt' vào trong đó.")
        return

    print(f"⏳ Đang đọc dữ liệu từ: {file_path}...")
    text = file_path.read_text(encoding="utf-8")
    
    # 1. Tạo thư mục logs nếu chưa có
    os.makedirs("logs", exist_ok=True)
    
    # 2. Thiết lập file log
    current_time = datetime.datetime.now()
    log_filename = f"logs/baseline_report.txt"

    # 3. Chạy thuật toán so sánh
    comparator = ChunkingStrategyComparator()
    results = comparator.compare(text, chunk_size=chunk_size)

    # 4. Ghi log và in ra màn hình
    with open(log_filename, "w", encoding="utf-8") as log_file:
        def lprint(msg=""):
            print(msg)
            log_file.write(msg + "\n")

        lprint("=" * 85)
        lprint(f" 📊 BÁO CÁO BASELINE ANALYSIS: CHUNKING STRATEGIES")
        lprint(f" Tài liệu            : {file_path.name} (~{len(text):,} ký tự)")
        lprint(f" Chunk Size          : {chunk_size}")
        lprint("=" * 85 + "\n")

        # In Header của bảng
        lprint(f"| {'Tài liệu':<12} | {'Strategy':<15} | {'Chunk Count':<12} | {'Avg Length':<12} | {'Preserves Context?':<25} |")
        lprint("|" + "-"*14 + "|" + "-"*17 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*27 + "|")

        # Duyệt qua các chiến lược và in kết quả
        for strategy, stats in results.items():
            count = stats["count"]
            avg_length = stats["avg_length"]
            
            # Đánh giá tự động hiển thị theo từng thuật toán
            if strategy == "fixed_size":
                eval_text = "Không — cắt ngang câu"
            elif strategy == "by_sentences":
                eval_text = "Tốt hơn — ngắt đúng câu"
            elif strategy == "recursive":
                eval_text = "Tốt nhất — ngắt theo đoạn"
            else:
                eval_text = "Chưa rõ"

            lprint(f"| {file_path.name:<12} | {strategy:<15} | {count:<12} | {avg_length:<12.1f} | {eval_text:<25} |")
            
        lprint(f"\n✅ Hoàn tất chạy Baseline! (Log được lưu tại: {log_filename})")

if __name__ == "__main__":
    # Đặt chunk_size=500 giống hệt với setting trong bảng báo cáo mẫu
    run_baseline(chunk_size=500)