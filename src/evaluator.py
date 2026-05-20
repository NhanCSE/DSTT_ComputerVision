"""
Bước 3 (phần 2): Đánh giá & So sánh hai phương pháp
=====================================================
Module này cung cấp:
  1. accuracy()            — tỉ lệ dự đoán đúng
  2. per_class_accuracy()  — accuracy theo từng người
  3. compare_k_values()    — Eigenfaces với nhiều giá trị k khác nhau
  4. run_full_comparison() — so sánh Eigenfaces vs Baseline, in báo cáo tổng hợp
"""

from __future__ import annotations

import time
import numpy as np


# ==============================================================================
# METRIC 1 — Accuracy tổng thể
# ==============================================================================
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tỉ lệ dự đoán đúng (Overall Accuracy).

    Công thức: accuracy = số_dự_đoán_đúng / tổng_số_mẫu

    Tham số:
        y_true : shape (N,) — nhãn thực.
        y_pred : shape (N,) — nhãn dự đoán.

    Trả về:
        float trong [0.0, 1.0].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # (y_true == y_pred) trả về mảng boolean True/False.
    # np.mean coi True=1, False=0 → trung bình chính là tỉ lệ True.
    return float(np.mean(y_true == y_pred))


# ==============================================================================
# METRIC 2 — Accuracy theo từng người (per-class)
# ==============================================================================
def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[int, float]:
    """
    Tính accuracy riêng cho từng person ID.

    Trả về:
        dict: {person_id: accuracy_của_người_đó}
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    # np.unique trả về các giá trị nhãn xuất hiện (không trùng).
    classes = np.unique(y_true)
    result  = {}
    for cls in classes:
        # mask: mảng boolean, True ở các vị trí có nhãn = cls.
        # y_pred[mask] = chỉ lấy các dự đoán ở những vị trí đó.
        mask         = y_true == cls
        result[cls]  = float(np.mean(y_pred[mask] == y_true[mask]))
    return result


# ==============================================================================
# THÍ NGHIỆM 1 — Eigenfaces với nhiều giá trị k
# ==============================================================================
def compare_k_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """
    Huấn luyện OrthogonalFaceRecognizer với nhiều giá trị k eigenfaces khác nhau,
    đo accuracy và thời gian, để tìm k tối ưu.

    Tham số:
        k_values : danh sách giá trị k muốn thử
                   (mặc định: [5, 10, 20, 30, 50, 75, 100, 150])

    Trả về:
        dict với keys: 'k_values', 'accuracies', 'train_times', 'predict_times'
    """
    from src.recognizer import OrthogonalFaceRecognizer

    if k_values is None:
        k_values = [5, 10, 20, 30, 50, 75, 100, 150]

    accuracies    = []
    train_times   = []
    predict_times = []

    print("\n" + "─" * 55)
    print(f"  {'k':>5}  {'Accuracy':>10}  {'Train(s)':>10}  {'Predict(s)':>12}")
    print("─" * 55)

    for k in k_values:
        # Huấn luyện
        t0 = time.perf_counter()
        rec = OrthogonalFaceRecognizer(n_components=k).fit(X_train, y_train)
        t_train = time.perf_counter() - t0

        # Dự đoán
        t0 = time.perf_counter()
        y_pred = rec.predict(X_test)
        t_pred = time.perf_counter() - t0

        acc = accuracy(y_test, y_pred)
        accuracies.append(acc)
        train_times.append(t_train)
        predict_times.append(t_pred)

        print(f"  {k:>5}  {acc*100:>9.1f}%  {t_train:>10.3f}  {t_pred:>12.3f}")

    print("─" * 55)

    best_idx = int(np.argmax(accuracies))
    print(f"  → k tốt nhất: {k_values[best_idx]}  (Accuracy = {accuracies[best_idx]*100:.1f}%)\n")

    return {
        "k_values"     : k_values,
        "accuracies"   : accuracies,
        "train_times"  : train_times,
        "predict_times": predict_times,
        "best_k"       : k_values[best_idx],
        "best_accuracy": accuracies[best_idx],
    }


# ==============================================================================
# THÍ NGHIỆM 2 — So sánh toàn diện: Eigenfaces vs Baseline
# ==============================================================================
def run_full_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    best_k: int = 50,
) -> dict:
    """
    Chạy so sánh đầy đủ giữa Eigenfaces và Pixel-KNN Baseline.

    1. Chạy Eigenfaces với k = best_k
    2. Chạy Pixel-KNN (k_nn=1) làm Baseline
    3. In bảng so sánh chi tiết
    4. Trả về kết quả để vẽ biểu đồ (Bước 4)

    Trả về:
        dict chứa kết quả của cả hai phương pháp.
    """
    from src.recognizer import OrthogonalFaceRecognizer
    from src.baseline   import PixelKNNRecognizer

    results = {}

    # ------------------------------------------------------------------
    # PHƯƠNG PHÁP 1: Eigenfaces (phép chiếu vuông góc)
    # ------------------------------------------------------------------
    print("Đang chạy Eigenfaces (Orthogonal Projection)...")
    t0 = time.perf_counter()
    eigen_rec = OrthogonalFaceRecognizer(n_components=best_k).fit(X_train, y_train)
    t_eigen_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_eigen = eigen_rec.predict(X_test)
    t_eigen_pred = time.perf_counter() - t0

    acc_eigen       = accuracy(y_test, y_pred_eigen)
    per_cls_eigen   = per_class_accuracy(y_test, y_pred_eigen)
    worst_cls_eigen = min(per_cls_eigen, key=per_cls_eigen.get)

    results["eigenfaces"] = {
        "y_pred"         : y_pred_eigen,
        "accuracy"       : acc_eigen,
        "per_class_acc"  : per_cls_eigen,
        "train_time"     : t_eigen_train,
        "predict_time"   : t_eigen_pred,
        "k"              : best_k,
        "recognizer"     : eigen_rec,
    }

    # ------------------------------------------------------------------
    # PHƯƠNG PHÁP 2: Pixel KNN Baseline (không chiếu)
    # ------------------------------------------------------------------
    print("Đang chạy Baseline (Pixel KNN)...")
    t0 = time.perf_counter()
    baseline_rec = PixelKNNRecognizer(k=1).fit(X_train, y_train)
    t_base_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_base = baseline_rec.predict(X_test)
    t_base_pred = time.perf_counter() - t0

    acc_base       = accuracy(y_test, y_pred_base)
    per_cls_base   = per_class_accuracy(y_test, y_pred_base)
    worst_cls_base = min(per_cls_base, key=per_cls_base.get)

    results["baseline"] = {
        "y_pred"        : y_pred_base,
        "accuracy"      : acc_base,
        "per_class_acc" : per_cls_base,
        "train_time"    : t_base_train,
        "predict_time"  : t_base_pred,
        "recognizer"    : baseline_rec,
    }

    # ------------------------------------------------------------------
    # IN BÁO CÁO SO SÁNH
    # ------------------------------------------------------------------
    _print_comparison_report(
        acc_eigen, acc_base,
        t_eigen_train, t_base_train,
        t_eigen_pred,  t_base_pred,
        best_k, X_train.shape[1],
        worst_cls_eigen, per_cls_eigen[worst_cls_eigen],
        worst_cls_base,  per_cls_base[worst_cls_base],
        y_test, y_pred_eigen, y_pred_base,
    )

    return results


def _print_comparison_report(
    acc_eigen, acc_base,
    t_eigen_train, t_base_train,
    t_eigen_pred,  t_base_pred,
    best_k, p,
    worst_eigen, worst_eigen_acc,
    worst_base,  worst_base_acc,
    y_test, y_pred_eigen, y_pred_base,
) -> None:
    """In bảng kết quả so sánh chi tiết ra console."""

    delta_acc  = (acc_eigen - acc_base) * 100
    speedup    = t_base_pred / t_eigen_pred if t_eigen_pred > 0 else float("inf")
    compress   = p / best_k  # hệ số nén không gian

    print("\n" + "=" * 60)
    print("  BÁO CÁO SO SÁNH: EIGENFACES vs BASELINE")
    print("=" * 60)
    print(f"  {'Tiêu chí':<32} {'Eigenfaces':>12} {'Baseline':>12}")
    print("─" * 60)
    print(f"  {'Accuracy':<32} {acc_eigen*100:>11.1f}% {acc_base*100:>11.1f}%")
    print(f"  {'Thời gian huấn luyện (s)':<32} {t_eigen_train:>12.3f} {t_base_train:>12.3f}")
    print(f"  {'Thời gian dự đoán (s)':<32} {t_eigen_pred:>12.3f} {t_base_pred:>12.3f}")
    print(f"  {'Không gian so sánh (chiều)':<32} {best_k:>12,} {p:>12,}")
    print("─" * 60)
    print(f"  Eigenfaces cải thiện accuracy  : {delta_acc:+.1f}%")
    print(f"  Eigenfaces nhanh hơn (predict) : {speedup:.1f}x")
    print(f"  Hệ số nén không gian            : {compress:.0f}x  ({p:,}D → {best_k}D)")
    print("─" * 60)

    # Phân tích sai lầm
    errors_eigen = np.sum(y_pred_eigen != y_test)
    errors_base  = np.sum(y_pred_base  != y_test)
    n_test       = len(y_test)

    print(f"  Số ảnh nhận dạng sai — Eigenfaces : {errors_eigen}/{n_test}")
    print(f"  Số ảnh nhận dạng sai — Baseline   : {errors_base}/{n_test}")
    print(f"  Người khó nhận nhất (Eigenfaces)   : person {worst_eigen} ({worst_eigen_acc*100:.0f}%)")
    print(f"  Người khó nhận nhất (Baseline)     : person {worst_base} ({worst_base_acc*100:.0f}%)")
    print("=" * 60 + "\n")


# ==============================================================================
# CHẠY THỬ
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.dataloader import load_orl_dataset, download_orl_dataset

    data_dir = os.path.join(project_root, "data", "orl_faces")
    if not os.path.isdir(data_dir):
        download_orl_dataset(save_dir=os.path.join(project_root, "data"))

    print("Đang đọc dataset...")
    X_train, y_train, X_test, y_test = load_orl_dataset(data_dir=data_dir)

    # Thí nghiệm 1: tìm k tối ưu
    print("\n== THÍ NGHIỆM 1: Eigenfaces với nhiều giá trị k ==")
    k_results = compare_k_values(X_train, y_train, X_test, y_test)
    best_k    = k_results["best_k"]

    # Thí nghiệm 2: so sánh đầy đủ với k tốt nhất
    print("\n== THÍ NGHIỆM 2: So sánh Eigenfaces vs Baseline ==")
    results = run_full_comparison(X_train, y_train, X_test, y_test, best_k=best_k)

    print("Bước 3 hoàn tất. Sẵn sàng cho Bước 4 (Visualization).")
