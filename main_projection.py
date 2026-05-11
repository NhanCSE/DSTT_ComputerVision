"""
main_projection.py — Entry point toàn bộ dự án
================================================
Chạy file này để thực hiện đầy đủ pipeline từ Bước 1 đến Bước 6:

    python main_projection.py

Kết quả:
    - In báo cáo đánh giá ra console
    - Lưu toàn bộ biểu đồ vào thư mục outputs/

Cấu trúc pipeline:
    Bước 1 — Tải và vector hóa dataset ORL Faces
    Bước 2 — Huấn luyện OrthogonalFaceRecognizer (Eigenfaces)
    Bước 3 — Tìm k tối ưu + so sánh với Baseline Pixel-KNN
    Bước 4 — Sinh biểu đồ cho báo cáo và slide
    Bước 6 — Ứng dụng mở rộng: tái tạo ảnh và khử nhiễu
    (Bước 5 được minh họa riêng: chạy  python manual_example.py)
"""

import os
import sys
import time

import numpy as np

# Thêm thư mục gốc vào sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.dataloader  import (download_orl_dataset, load_orl_dataset,
                              print_dataset_info, IMG_HEIGHT, IMG_WIDTH)
from src.recognizer  import OrthogonalFaceRecognizer
from src.baseline    import PixelKNNRecognizer
from src.evaluator   import (compare_k_values, run_full_comparison, accuracy)
from src.visualizer  import plot_all
from extended_applications import run_extended_applications


# ==============================================================================
# CẤU HÌNH — Thay đổi các thông số ở đây nếu cần
# ==============================================================================

DATA_DIR    = os.path.join(_ROOT, "data", "orl_faces")
OUTPUT_DIR  = os.path.join(_ROOT, "outputs")

# Danh sách giá trị k để thử trong Bước 3
K_VALUES_TO_TRY = [5, 10, 20, 30, 50, 75, 100, 150]

# Số người và số ảnh/người trong ORL dataset
N_PERSONS           = 40
N_IMAGES_PER_PERSON = 10
N_TEST_PER_PERSON   = 2   # 2 ảnh cuối mỗi người → tập test


# ==============================================================================
# TIỆN ÍCH
# ==============================================================================

def _section(title: str) -> None:
    """In tiêu đề phân cách các bước."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.2f}s"


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main() -> None:
    t_total = time.perf_counter()
    print("\n" + "=" * 60)
    print("  NHẬN DẠNG KHUÔN MẶT BẰNG PHÉP CHIẾU VUÔNG GÓC")
    print("  (Eigenfaces — Turk & Pentland, 1991)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # BƯỚC 1 — Tải & vector hóa dataset
    # ------------------------------------------------------------------
    _section("BƯỚC 1 — Chuẩn bị Dataset")
    t0 = time.perf_counter()

    if not os.path.isdir(DATA_DIR):
        download_orl_dataset(save_dir=os.path.join(_ROOT, "data"))

    X_train, y_train, X_test, y_test = load_orl_dataset(
        data_dir           = DATA_DIR,
        n_persons          = N_PERSONS,
        n_images_per_person= N_IMAGES_PER_PERSON,
        n_test_per_person  = N_TEST_PER_PERSON,
    )
    print_dataset_info(X_train, y_train, X_test, y_test)
    print(f"  Thời gian: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # BƯỚC 3 (chạy trước Bước 2) — Tìm k tối ưu
    # ------------------------------------------------------------------
    _section("BƯỚC 3a — Tìm số Eigenfaces k tối ưu")
    t0 = time.perf_counter()

    k_results = compare_k_values(
        X_train, y_train, X_test, y_test,
        k_values=K_VALUES_TO_TRY,
    )
    best_k = k_results["best_k"]
    print(f"  → Chọn k = {best_k}  |  Thời gian: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # BƯỚC 2 — Huấn luyện mô hình Eigenfaces với k tối ưu
    # ------------------------------------------------------------------
    _section(f"BƯỚC 2 — Huấn luyện OrthogonalFaceRecognizer  (k = {best_k})")
    t0 = time.perf_counter()

    recognizer = OrthogonalFaceRecognizer(n_components=best_k)
    recognizer.fit(X_train, y_train)
    recognizer.print_summary()
    print(f"  Thời gian huấn luyện: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # BƯỚC 3b — So sánh đầy đủ Eigenfaces vs Baseline
    # ------------------------------------------------------------------
    _section("BƯỚC 3b — So sánh Eigenfaces vs Baseline Pixel-KNN")
    t0 = time.perf_counter()

    comparison = run_full_comparison(
        X_train, y_train, X_test, y_test,
        best_k=best_k,
    )
    baseline_acc = comparison["baseline"]["accuracy"]
    eigen_acc    = comparison["eigenfaces"]["accuracy"]
    print(f"  Thời gian: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # BƯỚC 4 — Sinh toàn bộ biểu đồ
    # ------------------------------------------------------------------
    _section("BƯỚC 4 — Trực quan hóa")
    t0 = time.perf_counter()

    plot_all(
        recognizer        = recognizer,
        X_train           = X_train,
        y_train           = y_train,
        X_test            = X_test,
        y_test            = y_test,
        k_results         = k_results,
        baseline_accuracy = baseline_acc,
        output_dir        = OUTPUT_DIR,
    )
    print(f"  Thời gian: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # BƯỚC 6 — Ứng dụng mở rộng (dùng recognizer với k lớn hơn để đủ eigenfaces)
    # ------------------------------------------------------------------
    _section("BƯỚC 6 — Ứng dụng mở rộng: Tái tạo & Khử nhiễu")
    t0 = time.perf_counter()

    # Huấn luyện lại với k=150 để có đủ eigenfaces cho mọi thí nghiệm tái tạo
    k_extended = min(150, X_train.shape[0] - 1)
    print(f"  Huấn luyện lại với k = {k_extended} cho ứng dụng mở rộng...")
    rec_extended = OrthogonalFaceRecognizer(n_components=k_extended)
    rec_extended.fit(X_train, y_train)

    run_extended_applications(
        recognizer = rec_extended,
        X_test     = X_test,
        y_test     = y_test,
        img_shape  = (IMG_HEIGHT, IMG_WIDTH),
        output_dir = OUTPUT_DIR,
    )
    print(f"  Thời gian: {_elapsed(t0)}")

    # ------------------------------------------------------------------
    # TÓM TẮT CUỐI
    # ------------------------------------------------------------------
    _section("TÓM TẮT KẾT QUẢ")
    print(f"""
  Phương pháp                : Eigenfaces (Phép chiếu vuông góc)
  Số Eigenfaces tối ưu (k)   : {best_k}
  Accuracy — Eigenfaces      : {eigen_acc*100:.1f}%
  Accuracy — Baseline KNN    : {baseline_acc*100:.1f}%
  Cải thiện                  : {(eigen_acc - baseline_acc)*100:+.1f}%

  Biểu đồ đã lưu vào        : {OUTPUT_DIR}/
    ├── mean_face.png
    ├── eigenfaces.png
    ├── recognition.png
    ├── accuracy_vs_k.png
    ├── variance_ratio.png
    ├── app1_reconstruction.png
    ├── app1_quality_curve.png
    ├── app2_denoising.png
    └── app2_psnr_gain.png

  Ví dụ tính tay (Bước 5)   : python manual_example.py
  Tổng thời gian pipeline    : {_elapsed(t_total)}
    """)


# ==============================================================================

if __name__ == "__main__":
    main()
