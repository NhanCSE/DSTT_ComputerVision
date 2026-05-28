"""
Bước 2: Hiện thực thuật toán Core — Eigenfaces với Phép chiếu vuông góc
=========================================================================
Lớp OrthogonalFaceRecognizer hiện thực phương pháp Eigenfaces (Turk & Pentland, 1991).

Luồng thuật toán:
  fit(X_train):
    1. Tính khuôn mặt trung bình x̄
    2. Trung tâm hóa: Φi = xi - x̄
    3. Tính ma trận hiệp phương sai (dùng thủ thuật giảm chiều)
    4. Phân rã trị riêng → các eigenfaces U
    5. Chiếu tập train xuống không gian eigenface (lưu lại để nhận dạng)

  project(X):
    ŷ = U^T (x - x̄)   ← phép chiếu vuông góc cốt lõi

  predict(X_test):
    Nhận dạng bằng 1-Nearest Neighbor trên không gian eigenface

Ràng buộc:
  - KHÔNG dùng sklearn, scipy hay bất kỳ hàm "hộp đen" nào.
  - Tất cả đại số tuyến tính đều dùng numpy thuần.

Tài liệu tham khảo:
  [1] Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition."
      Journal of Cognitive Neuroscience, 3(1), 71–86.
  [2] NumPy documentation — numpy.linalg.eigh:
      https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
"""

from __future__ import annotations  # cho phép dùng 'int | None' trên Python 3.8+

import numpy as np


class OrthogonalFaceRecognizer:
    """
    Nhận dạng khuôn mặt bằng phép chiếu vuông góc lên không gian Eigenface.

    Tham số khởi tạo:
        n_components (int | None):
            Số eigenfaces muốn giữ lại (k).
            Nếu None, giữ tất cả eigenvectors hợp lệ.
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

        # Các thuộc tính được điền sau khi gọi fit()
        self.mean_face_         : np.ndarray | None = None  # x̄, shape (p,)
        self.eigenfaces_        : np.ndarray | None = None  # U, shape (p, k)
        self.eigenvalues_       : np.ndarray | None = None  # λ, shape (k,) giảm dần
        self.train_projections_ : np.ndarray | None = None  # shape (N_train, k)
        self.train_labels_      : np.ndarray | None = None  # shape (N_train,)

    # ==========================================================================
    # FIT — Học không gian Eigenface từ tập huấn luyện
    # ==========================================================================
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "OrthogonalFaceRecognizer":
        """
        Xây dựng không gian Eigenface từ tập ảnh huấn luyện.

        Tham số:
            X_train : shape (N, p) — N ảnh, mỗi ảnh là vector p chiều.
            y_train : shape (N,)   — nhãn person ID tương ứng.

        Trả về:
            self (để hỗ trợ cú pháp chaining: recognizer.fit(...).predict(...))
        """
        # ------------------------------------------------------------------
        # Quy ước shape:
        #   X_train : (N, p)
        #     - mỗi HÀNG là một ảnh đã flatten thành vector p chiều.
        #     - N = số ảnh train (vd: 320), p = số pixel (vd: 10304).
        #
        # Lưu ý cho người mới:
        #   - Trong nhiều sách toán, ảnh được xếp theo CỘT (X có shape (p, N)).
        #     Ở đây chúng ta dùng theo HÀNG vì numpy thao tác hàng tiện hơn.
        #   - Do đó các công thức bên dưới có thể khác sách một chút
        #     (vd: dùng Phi @ Phi.T thay vì Phi.T @ Phi).
        # ------------------------------------------------------------------
        N, p = X_train.shape  # N = số ảnh train, p = số pixel mỗi ảnh

        # ------------------------------------------------------------------
        # BƯỚC 2.1 — Khuôn mặt trung bình (Mean Face)
        #
        # x̄ = (1/N) Σ xi   (trung bình cộng theo từng pixel)
        # np.mean(..., axis=0) tính trung bình dọc theo chiều ảnh (hàng).
        # ------------------------------------------------------------------
        self.mean_face_ = np.mean(X_train, axis=0)   # shape: (p,)

        # ------------------------------------------------------------------
        # BƯỚC 2.2 — Trung tâm hóa dữ liệu
        #
        # Φi = xi - x̄  (ảnh sai lệch so với khuôn mặt trung bình)
        # Broadcasting: numpy tự trừ mean_face_ khỏi mỗi hàng của X_train.
        # ------------------------------------------------------------------
        Phi = X_train - self.mean_face_               # shape: (N, p)

        # ------------------------------------------------------------------
        # BƯỚC 2.3 — Ma trận hiệp phương sai (Covariance Matrix)
        #
        # Cách trực tiếp:  C = Phi^T @ Phi / N   →  shape (p, p) = (10304, 10304)
        #   → Chiếm ~800MB RAM và phân rã trị riêng cực chậm.
        #
        # Thủ thuật Turk & Pentland [1]:
        #   Tính L = Phi @ Phi^T / N   →  shape (N, N) = (320, 320) — nhỏ hơn nhiều!
        #
        #   Tại sao đúng?
        #   Giả sử L·v = λ·v  (v là eigenvector của L).
        #   Nhân trái bởi Phi^T:  Phi^T·L·v = λ·(Phi^T·v)
        #   ⟺  (Phi^T·Phi/N)·(Phi^T·v) = λ·(Phi^T·v)
        #   ⟺  C·u = λ·u   với u = Phi^T·v
        #   → u = Phi^T·v là eigenvector của C với cùng eigenvalue λ!  ✓
        # ------------------------------------------------------------------
        L = (Phi @ Phi.T) / N                         # shape: (N, N)

        # ------------------------------------------------------------------
        # BƯỚC 2.4 — Phân rã trị riêng của L
        #
        # np.linalg.eigh() dùng cho ma trận THỰC ĐỐI XỨNG (symmetric):
        #   - Ổn định số học hơn np.linalg.eig() cho trường hợp này.
        #   - Đảm bảo trị riêng là số thực.
        #   - Trả về eigenvalues TĂNG DẦN → cần đảo ngược.
        #
        # Nguồn: NumPy docs [2].
        # ------------------------------------------------------------------
        eigenvalues_L, V = np.linalg.eigh(L)
        # eigenvalues_L: shape (N,) — trị riêng của L, tăng dần
        # V            : shape (N, N) — mỗi CỘT là một eigenvector của L

        # Đảo ngược để có thứ tự GIẢM DẦN (eigenvalue lớn = giải thích nhiều variance hơn)
        idx            = np.argsort(eigenvalues_L)[::-1]
        eigenvalues_L  = eigenvalues_L[idx]           # shape: (N,)
        V              = V[:, idx]                    # shape: (N, N)

        # Loại bỏ eigenvalue âm hoặc gần 0 (sai số số học)
        valid          = eigenvalues_L > 1e-10
        eigenvalues_L  = eigenvalues_L[valid]         # shape: (n_valid,)
        V              = V[:, valid]                  # shape: (N, n_valid)

        # ------------------------------------------------------------------
        # BƯỚC 2.5 — Khôi phục Eigenvectors của C (các Eigenfaces)
        #
        # u_i = Phi^T @ v_i / ||Phi^T @ v_i||
        #
        # Phép tính ma trận một lần: Phi.T @ V cho tất cả eigenvectors cùng lúc.
        # ------------------------------------------------------------------
        U_unnorm = Phi.T @ V                          # shape: (p, n_valid)

        # Chuẩn hóa mỗi cột về độ dài 1 (unit norm)
        norms = np.linalg.norm(U_unnorm, axis=0, keepdims=True)  # shape: (1, n_valid)
        norms = np.where(norms < 1e-10, 1.0, norms)  # tránh chia cho 0
        U     = U_unnorm / norms                      # shape: (p, n_valid)

        # ------------------------------------------------------------------
        # BƯỚC 2.6 — Chọn k eigenfaces tốt nhất
        # ------------------------------------------------------------------
        n_valid = U.shape[1]
        k = self.n_components if self.n_components is not None else n_valid
        k = min(k, n_valid)

        self.eigenvalues_ = eigenvalues_L[:k]         # shape: (k,)
        self.eigenfaces_  = U[:, :k]                  # shape: (p, k)

        # ------------------------------------------------------------------
        # BƯỚC 2.7 — Chiếu tập train (lưu để dùng khi predict)
        # ------------------------------------------------------------------
        self.train_projections_ = self.project(X_train)  # shape: (N, k)
        self.train_labels_      = y_train.copy()

        return self

    # ==========================================================================
    # PROJECT — Phép chiếu vuông góc (trọng tâm toán học của bài)
    # ==========================================================================
    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Chiếu tập ảnh X xuống không gian con Eigenface.

        Công thức:  ŷ = U^T (x - x̄)

        Giải thích hình học:
          - (x - x̄) là vector ảnh đã trung tâm hóa, nằm trong R^p.
          - U là ma trận có các cột trực chuẩn (orthonormal), mỗi cột là một eigenface.
          - U^T (x - x̄) là tọa độ của (x - x̄) trong hệ cơ sở {u1, u2, ..., uk}.
          - Đây chính xác là phép chiếu vuông góc lên không gian con span(u1,...,uk).

        Tham số:
            X : shape (N, p) hoặc (p,) — ảnh cần chiếu.

        Trả về:
            shape (N, k) hoặc (k,) — tọa độ trong không gian Eigenface.
        """
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Trung tâm hóa: x - x̄
        # Broadcasting: numpy tự trừ mean_face_ khỏi MỖI hàng của X.
        X_centered   = X - self.mean_face_             # shape: (N, p)

        # Chiếu xuống không gian eigenface.
        # Giải thích: với một ảnh đơn (vector hàng 1×p):
        #     (1×p) @ (p×k) = (1×k)   ← tọa độ k chiều
        # Tương đương công thức toán U^T·(x - x̄), nhưng vì X được xếp theo HÀNG
        # nên ta nhân (X - x̄) @ U thay vì U^T·(X - x̄).
        projections  = X_centered @ self.eigenfaces_   # shape: (N, k)

        return projections[0] if single else projections

    # ==========================================================================
    # PREDICT — Nhận dạng bằng 1-Nearest Neighbor
    # ==========================================================================
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Nhận dạng từng ảnh trong X_test bằng khoảng cách Euclidean (1-NN).

        Thuật toán:
          1. Chiếu ảnh test: ŷ_test = U^T (x_test - x̄)
          2. Với mỗi ŷ_test, tính khoảng cách tới tất cả ŷ_train
          3. Nhãn = nhãn của ảnh train gần nhất

        Tham số:
            X_test : shape (N_test, p) hoặc (p,).

        Trả về:
            shape (N_test,) — nhãn dự đoán (person ID).
        """
        single = X_test.ndim == 1
        if single:
            X_test = X_test.reshape(1, -1)

        # Bước 1: Chiếu ảnh test
        test_projs = self.project(X_test)              # shape: (N_test, k)

        N_test      = test_projs.shape[0]
        predictions = np.zeros(N_test, dtype=np.int32)

        for i in range(N_test):
            # Bước 2: Khoảng cách Euclidean từ ảnh test thứ i tới mỗi ảnh train
            # Công thức: ||ŷ_test - ŷ_train_j||_2 = sqrt(Σ (ŷ_test_l - ŷ_train_jl)^2)
            #
            # Cách tính vector hoá (nhanh hơn vòng for):
            #   diffs[j] = train_projections_[j] - test_projs[i]   (vector k chiều)
            #   distance[j] = sqrt(sum(diffs[j] ** 2))             (số thực)
            # Nhờ broadcasting, numpy tính được TẤT CẢ N_train khoảng cách
            # cùng một lúc — không cần vòng for thứ hai.
            diffs     = self.train_projections_ - test_projs[i]  # shape: (N_train, k)
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))      # shape: (N_train,)

            # Bước 3: Chọn ảnh train có khoảng cách nhỏ nhất (1-Nearest Neighbor).
            # → Nhãn dự đoán = nhãn của ảnh train đó.
            nearest_idx   = np.argmin(distances)
            predictions[i] = self.train_labels_[nearest_idx]

        return int(predictions[0]) if single else predictions

    # ==========================================================================
    # TIỆN ÍCH — Phân tích phương sai
    # ==========================================================================
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Tỉ lệ phương sai được giải thích bởi từng eigenface.

        Eigenvalue λ_i tỉ lệ thuận với phương sai theo hướng eigenface u_i.
        ratio_i = λ_i / Σλ

        Trả về:
            shape (k,) — ratio_i trong [0, 1], tổng = 1.
        """
        total = np.sum(self.eigenvalues_)
        return self.eigenvalues_ / total

    def cumulative_explained_variance(self) -> np.ndarray:
        """Phương sai tích lũy: cumvar[i] = Σ_{j=0}^{i} ratio_j."""
        return np.cumsum(self.explained_variance_ratio())

    def n_components_for_variance(self, target: float = 0.95) -> int:
        """
        Số eigenfaces tối thiểu cần để giải thích ít nhất `target` phương sai.
        Ví dụ: n_components_for_variance(0.95) → k sao cho top-k giải thích 95%.
        """
        cumvar = self.cumulative_explained_variance()
        k = int(np.searchsorted(cumvar, target)) + 1
        return min(k, len(self.eigenvalues_))

    def print_summary(self) -> None:
        """In tóm tắt mô hình đã huấn luyện."""
        if self.eigenfaces_ is None:
            print("Mô hình chưa được huấn luyện (chưa gọi fit()).")
            return

        p, k    = self.eigenfaces_.shape
        n_train = len(self.train_labels_)
        var_1   = self.explained_variance_ratio()[0] * 100
        var_k   = self.cumulative_explained_variance()[-1] * 100
        k_95    = self.n_components_for_variance(0.95)

        print("\n" + "=" * 52)
        print("  TÓM TẮT MÔ HÌNH EIGENFACES")
        print("=" * 52)
        print(f"  Không gian ảnh gốc (p)         : {p:,} chiều")
        print(f"  Số ảnh huấn luyện (N)          : {n_train}")
        print(f"  Số Eigenfaces giữ lại (k)       : {k}")
        print(f"  Eigenvalue lớn nhất (λ₁)        : {self.eigenvalues_[0]:.2f}")
        print(f"  Eigenvalue nhỏ nhất giữ (λ_k)   : {self.eigenvalues_[-1]:.4f}")
        print(f"  Eigenface #1 giải thích         : {var_1:.1f}% phương sai")
        print(f"  Top-{k} eigenfaces giải thích    : {var_k:.1f}% phương sai")
        print(f"  Cần bao nhiêu k để đạt 95%      : {k_95} eigenfaces")
        print("=" * 52 + "\n")


# ==============================================================================
# CHẠY THỬ (demo khi chạy file này trực tiếp)
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys

    # Thêm thư mục gốc vào sys.path để import src.dataloader
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.dataloader import load_orl_dataset, download_orl_dataset, print_dataset_info

    # Tải dataset
    data_dir = os.path.join(project_root, "data", "orl_faces")
    if not os.path.isdir(data_dir):
        download_orl_dataset(save_dir=os.path.join(project_root, "data"))

    print("Đang đọc dataset...")
    X_train, y_train, X_test, y_test = load_orl_dataset(data_dir=data_dir)
    print_dataset_info(X_train, y_train, X_test, y_test)

    # Huấn luyện mô hình với k=50 eigenfaces
    print("Đang huấn luyện OrthogonalFaceRecognizer (k=50)...")
    recognizer = OrthogonalFaceRecognizer(n_components=50)
    recognizer.fit(X_train, y_train)
    recognizer.print_summary()

    # Đánh giá trên tập test
    print("Đang nhận dạng tập test...")
    y_pred    = recognizer.predict(X_test)
    accuracy  = np.mean(y_pred == y_test) * 100
    n_correct = np.sum(y_pred == y_test)

    print(f"Kết quả: {n_correct}/{len(y_test)} đúng → Accuracy = {accuracy:.1f}%")
    print("\nBước 2 hoàn tất. Sẵn sàng cho Bước 3 (So sánh & Đánh giá).")
