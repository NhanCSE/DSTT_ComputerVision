"""
Bước 3 (phần 1): Baseline — K-Nearest Neighbors trong không gian pixel
=======================================================================
Đây là phương pháp so sánh (mốc đối chiếu) KHÔNG dùng phép chiếu vuông góc.

Ý tưởng:
  Với mỗi ảnh test, tính khoảng cách Euclidean tới TẤT CẢ ảnh train
  trực tiếp trong không gian pixel (R^p, p = 10304 chiều).
  Nhãn = nhãn của k ảnh train gần nhất (bầu chọn đa số).

Điểm khác biệt so với Eigenfaces:
  - Eigenfaces: so sánh trong không gian R^k (k << p) sau khi chiếu
  - Baseline  : so sánh trực tiếp trong R^p — không nén, không chiếu
  → Baseline chậm hơn và kém chính xác hơn khi dữ liệu nhiễu hoặc ánh sáng thay đổi.

Tài liệu tham khảo:
  [1] Cover, T. & Hart, P. (1967). "Nearest Neighbor Pattern Classification."
      IEEE Transactions on Information Theory, 13(1), 21–27.
"""

from __future__ import annotations

import numpy as np


class PixelKNNRecognizer:
    """
    Nhận dạng khuôn mặt bằng K-Nearest Neighbors trực tiếp trên pixel.

    Không có bước học đặc trưng (feature learning) — toàn bộ tập train
    được lưu lại và dùng để so sánh trực tiếp lúc dự đoán.

    Tham số:
        k (int): số láng giềng gần nhất dùng để bầu chọn nhãn (mặc định 1).
    """

    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError("k phải >= 1.")
        self.k        = k
        self.X_train_ : np.ndarray | None = None  # ảnh train gốc, shape (N, p)
        self.y_train_ : np.ndarray | None = None  # nhãn train, shape (N,)

    # ==========================================================================
    # FIT — Lưu toàn bộ tập train (lazy learning)
    # ==========================================================================
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> PixelKNNRecognizer:
        """
        Lưu tập train. KNN không có bước "học" thực sự —
        toàn bộ công việc tính toán diễn ra lúc predict().

        Tham số:
            X_train : shape (N, p) — ảnh huấn luyện.
            y_train : shape (N,)   — nhãn person ID.
        """
        self.X_train_ = X_train.copy()   # lưu bản sao để tránh thay đổi ngoài
        self.y_train_ = y_train.copy()
        return self

    # ==========================================================================
    # PREDICT — Dự đoán bằng khoảng cách Euclidean trong không gian pixel
    # ==========================================================================
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Nhận dạng từng ảnh trong X_test bằng k-NN pixel.

        Thuật toán:
          1. Tính khoảng cách Euclidean từ ảnh test tới MỌI ảnh train
             (trực tiếp trong R^p, KHÔNG chiếu).
          2. Chọn k ảnh train gần nhất.
          3. Bầu chọn đa số (majority vote) → nhãn dự đoán.

        Tham số:
            X_test : shape (N_test, p) hoặc (p,).

        Trả về:
            shape (N_test,) — nhãn dự đoán.
        """
        single = X_test.ndim == 1
        if single:
            X_test = X_test.reshape(1, -1)

        N_test      = X_test.shape[0]
        predictions = np.zeros(N_test, dtype=np.int32)

        for i in range(N_test):
            # Bước 1: Khoảng cách Euclidean tới tất cả ảnh train
            # ||x_test - x_train_j||_2 = sqrt(Σ (x_test_l - x_train_jl)^2)
            diffs     = self.X_train_ - X_test[i]                # shape: (N_train, p)
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))      # shape: (N_train,)

            # Bước 2: Chỉ số k ảnh train gần nhất
            k_nearest_idx    = np.argsort(distances)[: self.k]   # shape: (k,)
            k_nearest_labels = self.y_train_[k_nearest_idx]      # shape: (k,)

            # Bước 3: Majority vote — nhãn xuất hiện nhiều nhất trong k láng giềng
            # np.bincount đếm tần suất từng nhãn (chỉ dùng được với nhãn nguyên dương)
            vote_counts      = np.bincount(k_nearest_labels)
            predictions[i]   = np.argmax(vote_counts)

        return int(predictions[0]) if single else predictions

    def print_summary(self) -> None:
        """In tóm tắt cấu hình Baseline."""
        n_train = len(self.y_train_) if self.y_train_ is not None else 0
        p       = self.X_train_.shape[1] if self.X_train_ is not None else 0
        print("\n" + "=" * 52)
        print("  TÓM TẮT MÔ HÌNH BASELINE (Pixel KNN)")
        print("=" * 52)
        print(f"  Số ảnh huấn luyện (N)  : {n_train}")
        print(f"  Không gian so sánh (p) : {p:,} chiều  ← KHÔNG chiếu")
        print(f"  Số láng giềng (k)      : {self.k}")
        print("=" * 52 + "\n")
