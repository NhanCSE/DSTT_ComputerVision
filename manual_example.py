"""
Bước 5: Ví dụ minh họa tính tay (Manual Worked Example)
=========================================================
Mục đích: minh họa TỪNG BƯỚC thuật toán Eigenfaces trên tập dữ liệu nhỏ
(4 ảnh 3×3 pixel = vector 9 chiều) để người đọc báo cáo có thể đối chiếu
với phần lý thuyết và kiểm tra thủ công bằng máy tính cầm tay.

Cấu trúc dữ liệu đồ chơi:
  - 4 ảnh huấn luyện:  2 ảnh cho Person 1, 2 ảnh cho Person 2
  - 1 ảnh test:        gần với Person 1 → kỳ vọng nhận dạng đúng Person 1

  Person 1 — khuôn mặt "tối" (gradient tăng dần từ trái sang phải)
  Person 2 — khuôn mặt "sáng" (gradient giảm dần từ trái sang phải)

Tài liệu tham khảo:
  [1] Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition."
      Journal of Cognitive Neuroscience, 3(1), 71–86.
  [2] Strang, G. (2016). "Introduction to Linear Algebra." 5th ed. Wellesley.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=100)


# ==============================================================================
# TIỆN ÍCH IN ẤN
# ==============================================================================

def _header(step: str, formula: str = "") -> None:
    """In tiêu đề bước với công thức toán học tương ứng."""
    print("\n" + "=" * 65)
    print(f"  {step}")
    if formula:
        print(f"  Công thức: {formula}")
    print("=" * 65)


def _subheader(text: str) -> None:
    print(f"\n  ── {text}")


def _mat(name: str, M: np.ndarray, unit: str = "") -> None:
    """In một ma trận/vector với tên và kích thước."""
    shape_str = f"shape {M.shape}" if M.ndim > 1 else f"shape ({len(M)},)"
    print(f"\n  {name}  [{shape_str}]{('  ' + unit) if unit else ''}:")
    if M.ndim == 1:
        # Vector: in thành 1 hàng
        print("  ", np.array2string(M, separator=",  ", prefix="  "))
    else:
        for row in M:
            print("  ", np.array2string(row, separator=",  ", prefix="  "))


# ==============================================================================
# DỮ LIỆU ĐỒ CHƠI
# ==============================================================================

def define_dataset():
    """
    Định nghĩa tập dữ liệu nhỏ.

    Mỗi ảnh 3×3 pixel được "duỗi" thành vector 9 chiều.
    Dữ liệu được chọn để sự khác biệt giữa 2 người rõ ràng,
    nhưng vẫn có biến động nhỏ trong từng người.

    Bố cục pixel ảnh 3×3 (theo thứ tự flatten):
        [p0, p1, p2]
        [p3, p4, p5]
        [p6, p7, p8]
    """
    # fmt: off
    X_train = np.array([
        # p0   p1   p2   p3   p4   p5   p6   p7   p8
        [ 10,  20,  30,  20,  30,  40,  30,  40,  50],  # Person 1, ảnh 1
        [ 12,  22,  28,  22,  28,  42,  28,  42,  48],  # Person 1, ảnh 2
        [ 90,  80,  70,  80,  70,  60,  70,  60,  50],  # Person 2, ảnh 1
        [ 88,  78,  72,  78,  72,  58,  72,  58,  52],  # Person 2, ảnh 2
    ], dtype=float)
    # fmt: on

    y_train = np.array([1, 1, 2, 2])  # nhãn: 1 = Person 1, 2 = Person 2

    # Ảnh test: gần với Person 1 (giá trị pixel tăng dần)
    x_test  = np.array([11, 21, 29, 21, 29, 41, 29, 41, 49], dtype=float)
    y_true  = 1  # kỳ vọng: nhận dạng ra Person 1

    return X_train, y_train, x_test, y_true


# ==============================================================================
# BƯỚC 1 — Hiển thị dữ liệu đầu vào
# ==============================================================================

def step1_show_data(X_train, y_train, x_test):
    _header(
        "BƯỚC 1 — Dữ liệu đầu vào",
        "A = [x₁ | x₂ | ... | xₙ]  (N ảnh, mỗi ảnh là vector p chiều)",
    )
    N, p = X_train.shape
    print(f"\n  Số ảnh huấn luyện : N = {N}")
    print(f"  Kích thước vector : p = {p}  (ảnh {int(p**0.5)}×{int(p**0.5)} pixel, flatten)")

    for i in range(N):
        label = f"x{i+1} (Person {y_train[i]})"
        _mat(label, X_train[i])

    _mat("x_test (cần nhận dạng)", x_test)


# ==============================================================================
# BƯỚC 2 — Khuôn mặt trung bình
# ==============================================================================

def step2_mean_face(X_train):
    _header(
        "BƯỚC 2 — Khuôn mặt trung bình (Mean Face)",
        "x̄ = (1/N) Σᵢ xᵢ",
    )
    N = X_train.shape[0]

    mean_face = np.mean(X_train, axis=0)

    _subheader("Tính từng thành phần:")
    print(f"\n  Pixel thứ j:  x̄ⱼ = (x₁ⱼ + x₂ⱼ + x₃ⱼ + x₄ⱼ) / {N}")
    for j in range(len(mean_face)):
        vals = X_train[:, j]
        print(f"    j={j}: ({' + '.join(str(int(v)) for v in vals)}) / {N}"
              f" = {vals.sum():.0f} / {N} = {mean_face[j]:.2f}")

    _mat("x̄  (mean face)", mean_face)
    return mean_face


# ==============================================================================
# BƯỚC 3 — Trung tâm hóa (tính Φ)
# ==============================================================================

def step3_center(X_train, mean_face):
    _header(
        "BƯỚC 3 — Trung tâm hóa dữ liệu",
        "Φᵢ = xᵢ − x̄   (ảnh sai lệch so với trung bình)",
    )
    Phi = X_train - mean_face   # broadcasting: trừ mean_face khỏi từng hàng

    for i in range(len(Phi)):
        _mat(f"Φ{i+1} = x{i+1} − x̄", Phi[i])

    _subheader("Ma trận Φ (mỗi HÀNG là một ảnh đã trung tâm hóa):")
    _mat("Φ", Phi)
    return Phi


# ==============================================================================
# BƯỚC 4 — Ma trận hiệp phương sai (thủ thuật Turk & Pentland)
# ==============================================================================

def step4_covariance(Phi):
    _header(
        "BƯỚC 4 — Ma trận hiệp phương sai (thủ thuật tính nhanh)",
        "L = Φ·Φᵀ / N   [thay vì C = Φᵀ·Φ / N để tránh ma trận p×p khổng lồ]",
    )
    N = Phi.shape[0]

    _subheader("Tại sao dùng L thay vì C?")
    print(f"""
    Cách thông thường : C = Φᵀ·Φ / N  →  kích thước {Phi.shape[1]}×{Phi.shape[1]}  (lớn!)
    Thủ thuật         : L = Φ·Φᵀ / N  →  kích thước {N}×{N}         (nhỏ!)

    Tính chất: nếu L·v = λ·v  (v là eigenvector của L)
                thì  C·(Φᵀ·v) = λ·(Φᵀ·v)
                → Φᵀ·v là eigenvector của C với cùng eigenvalue λ.
    """)

    L = (Phi @ Phi.T) / N

    _subheader(f"Tính L = Φ·Φᵀ / {N}  (ma trận {N}×{N}):")
    print("\n  L[i,j] = (Φᵢ · Φⱼ) / N  =  tích vô hướng của ảnh i và ảnh j, chia N\n")

    for i in range(N):
        for j in range(i, N):
            dot = np.dot(Phi[i], Phi[j])
            print(f"    L[{i},{j}] = Φ{i+1}·Φ{j+1} / {N} = {dot:.2f} / {N} = {dot/N:.4f}")

    _mat("L  (surrogate covariance)", L)
    return L


# ==============================================================================
# BƯỚC 5 — Phân rã trị riêng của L
# ==============================================================================

def step5_eigendecomposition(L):
    _header(
        "BƯỚC 5 — Phân rã trị riêng của L",
        "L·V = V·Λ   →   eigenvalues λ và eigenvectors v",
    )
    _subheader("Dùng np.linalg.eigh() cho ma trận đối xứng:")
    print("""
    - eigh() ổn định hơn eig() cho ma trận thực đối xứng.
    - Trả về eigenvalues TĂNG DẦN → cần đảo ngược.
    """)

    eigenvalues_raw, V_raw = np.linalg.eigh(L)

    _mat("Eigenvalues (thứ tự tăng dần, gốc từ eigh)", eigenvalues_raw)
    _mat("V (mỗi CỘT là một eigenvector của L)", V_raw)

    # Đảo thứ tự: eigenvalue lớn nhất trước
    idx        = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[idx]
    V           = V_raw[:, idx]

    # Loại bỏ eigenvalue gần 0
    valid      = eigenvalues > 1e-10
    eigenvalues = eigenvalues[valid]
    V           = V[:, valid]

    _subheader("Sau khi sắp xếp GIẢM DẦN và loại eigenvalue ≈ 0:")
    _mat("Eigenvalues λ  (giảm dần)", eigenvalues)
    _mat("V  (eigenvectors của L, đã sắp xếp)", V)

    # Giải thích
    total = eigenvalues.sum()
    print("\n  Phương sai giải thích bởi từng eigenvector:")
    for i, lam in enumerate(eigenvalues):
        print(f"    v{i+1}: λ = {lam:.4f}  →  {lam/total*100:.1f}% phương sai")

    return eigenvalues, V


# ==============================================================================
# BƯỚC 6 — Khôi phục Eigenfaces (eigenvectors của C)
# ==============================================================================

def step6_recover_eigenfaces(Phi, V):
    _header(
        "BƯỚC 6 — Khôi phục Eigenfaces từ eigenvectors của L",
        "uᵢ = Φᵀ·vᵢ / ‖Φᵀ·vᵢ‖   (chuẩn hóa về unit vector)",
    )
    print("""
    Vì L·v = λ·v  nên  C·(Φᵀ·v) = λ·(Φᵀ·v)
    → u = Φᵀ·v  là eigenvector của C  (ma trận hiệp phương sai thực sự)
    → Cần chuẩn hóa u về độ dài 1 để tạo hệ cơ sở trực chuẩn.
    """)

    U_unnorm = Phi.T @ V           # shape: (p, n_valid)
    norms    = np.linalg.norm(U_unnorm, axis=0)

    for i in range(V.shape[1]):
        u_raw = Phi.T @ V[:, i]
        norm  = np.linalg.norm(u_raw)
        u_norm = u_raw / norm
        _subheader(f"Eigenface #{i+1}:")
        _mat(f"  Φᵀ·v{i+1}  (chưa chuẩn hóa)", u_raw)
        print(f"  ‖Φᵀ·v{i+1}‖ = {norm:.6f}")
        _mat(f"  u{i+1} = Φᵀ·v{i+1} / ‖Φᵀ·v{i+1}‖  (eigenface #{i+1})", u_norm)
        print(f"  Kiểm tra: ‖u{i+1}‖ = {np.linalg.norm(u_norm):.6f}  (phải = 1.0)")

    U = U_unnorm / norms
    _mat("\n  Ma trận U (mỗi CỘT là một eigenface)", U)

    # Kiểm tra tính trực giao
    _subheader("Kiểm tra tính trực chuẩn (orthonormality): Uᵀ·U phải = I")
    UtU = U.T @ U
    _mat("Uᵀ·U", UtU)

    return U


# ==============================================================================
# BƯỚC 7 — Chiếu ảnh train xuống không gian Eigenface
# ==============================================================================

def step7_project_train(X_train, y_train, mean_face, U):
    _header(
        "BƯỚC 7 — Chiếu ảnh train (lưu lại để nhận dạng)",
        "ŷᵢ = Uᵀ·(xᵢ − x̄)   (tọa độ trong không gian Eigenface)",
    )
    print("""
    Đây là phép chiếu vuông góc — trọng tâm của thuật toán.
    ŷᵢ là "danh thiếp" của ảnh xᵢ trong không gian Eigenface.
    Chỉ cần k chiều (k << p) để biểu diễn đặc trưng quan trọng nhất.
    """)

    train_projections = []
    for i in range(len(X_train)):
        centered = X_train[i] - mean_face
        proj     = U.T @ centered
        train_projections.append(proj)
        _mat(f"ŷ{i+1}  (Person {y_train[i]})", proj, "← tọa độ trong không gian eigenface")

    train_projections = np.array(train_projections)
    return train_projections


# ==============================================================================
# BƯỚC 8 — Chiếu ảnh test và nhận dạng
# ==============================================================================

def step8_recognize(x_test, y_true, mean_face, U, train_projections, y_train):
    _header(
        "BƯỚC 8 — Nhận dạng ảnh test",
        "ŷ_test = Uᵀ·(x_test − x̄),   nhận dạng = argmin ‖ŷ_test − ŷᵢ‖₂",
    )

    # Chiếu ảnh test
    x_centered = x_test - mean_face
    _mat("x_test − x̄  (trung tâm hóa ảnh test)", x_centered)

    y_hat = U.T @ x_centered
    _mat("ŷ_test = Uᵀ·(x_test − x̄)", y_hat, "← tọa độ ảnh test")

    # Tính khoảng cách Euclidean tới từng ảnh train
    _subheader("Khoảng cách Euclidean: d(ŷ_test, ŷᵢ) = ‖ŷ_test − ŷᵢ‖₂")
    print()
    distances = []
    for i, proj_i in enumerate(train_projections):
        diff = y_hat - proj_i
        dist = np.sqrt(np.sum(diff ** 2))
        distances.append(dist)
        print(f"    d(ŷ_test, ŷ{i+1}) = ‖{np.round(y_hat,3)} − {np.round(proj_i,3)}‖")
        print(f"              = ‖{np.round(diff,3)}‖  =  {dist:.4f}")
        print()

    distances = np.array(distances)

    # Nhận dạng: chọn ảnh train gần nhất
    nearest_idx    = int(np.argmin(distances))
    predicted_label = y_train[nearest_idx]

    _subheader("Kết quả:")
    print(f"    Khoảng cách nhỏ nhất: d{nearest_idx+1} = {distances[nearest_idx]:.4f}")
    print(f"    → Gán nhãn: Person {predicted_label}")
    if predicted_label == y_true:
        print(f"    ✓ ĐÚNG! (nhãn thực: Person {y_true})")
    else:
        print(f"    ✗ SAI.  (nhãn thực: Person {y_true})")

    return predicted_label, distances


# ==============================================================================
# BƯỚC 9 — Tóm tắt kết quả & hình ảnh hóa (matplotlib)
# ==============================================================================

def step9_visualize(X_train, y_train, x_test, mean_face, U, train_projections,
                    y_hat_test, distances):
    _header("BƯỚC 9 — Trực quan hóa ví dụ tính tay")

    try:
        import matplotlib.pyplot as plt
        import os
    except ImportError:
        print("  [SKIP] matplotlib chưa cài. Bỏ qua phần vẽ biểu đồ.")
        return

    IMG_SIZE = len(mean_face)
    side     = int(IMG_SIZE ** 0.5)  # chiều ảnh vuông (3 cho ảnh 3×3)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Ví dụ tính tay — Eigenfaces trên dữ liệu nhỏ (4 ảnh 3×3 pixel)",
                 fontsize=13, fontweight="bold")

    # ---- Hàng 1: các ảnh train + ảnh test ----
    n_train = len(X_train)
    for i in range(n_train):
        ax = fig.add_subplot(3, n_train + 2, i + 1)
        ax.imshow(X_train[i].reshape(side, side), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"x{i+1}\nPerson {y_train[i]}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(3, n_train + 2, n_train + 1)
    ax.imshow(mean_face.reshape(side, side), cmap="gray", vmin=0, vmax=255)
    ax.set_title("x̄\n(Mean Face)", fontsize=9, color="blue")
    ax.axis("off")

    ax = fig.add_subplot(3, n_train + 2, n_train + 2)
    ax.imshow(x_test.reshape(side, side), cmap="gray", vmin=0, vmax=255)
    ax.set_title("x_test\n(?)", fontsize=9, color="darkorange")
    ax.axis("off")

    # ---- Hàng 2: Eigenfaces (U[:,i] reshape) ----
    n_ef = U.shape[1]
    for i in range(n_ef):
        ax = fig.add_subplot(3, n_train + 2, (n_train + 2) + i + 1)
        ef  = U[:, i].reshape(side, side)
        ef_display = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
        ax.imshow(ef_display, cmap="gray")
        ax.set_title(f"Eigenface #{i+1}", fontsize=9, color="purple")
        ax.axis("off")

    # ---- Hàng 3: Không gian 2D eigenface + scatter plot ----
    ax = fig.add_subplot(3, 1, 3)
    colors = {1: "#3498db", 2: "#e74c3c"}
    for i, (proj, lbl) in enumerate(zip(train_projections, y_train)):
        ax.scatter(proj[0], proj[1], c=colors[lbl], s=120, zorder=3,
                   label=f"Person {lbl}" if i < 2 else "")
        ax.annotate(f"x{i+1}", (proj[0], proj[1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Ảnh test
    ax.scatter(y_hat_test[0], y_hat_test[1] if len(y_hat_test) > 1 else 0,
               c="orange", s=150, marker="*", zorder=4, label="x_test (truy vấn)")
    ax.annotate("x_test", (y_hat_test[0], y_hat_test[1] if len(y_hat_test) > 1 else 0),
                textcoords="offset points", xytext=(6, 4), fontsize=9, color="darkorange")

    ax.set_xlabel("Eigenface #1 (tọa độ 1)", fontsize=10)
    ax.set_ylabel("Eigenface #2 (tọa độ 2)" if U.shape[1] > 1 else "", fontsize=10)
    ax.set_title("Không gian Eigenface 2D — phép chiếu vuông góc", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/manual_example.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Đã lưu biểu đồ: {out_path}")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VÍ DỤ TÍNH TAY — EIGENFACES TRÊN DỮ LIỆU NHỎ")
    print("  (4 ảnh 3×3 pixel, 2 người, 1 ảnh test)")
    print("=" * 65)

    # Nạp dữ liệu
    X_train, y_train, x_test, y_true = define_dataset()

    # Thực hiện từng bước, in chi tiết
    step1_show_data(X_train, y_train, x_test)
    mean_face = step2_mean_face(X_train)
    Phi       = step3_center(X_train, mean_face)
    L         = step4_covariance(Phi)
    eigenvalues, V = step5_eigendecomposition(L)
    U         = step6_recover_eigenfaces(Phi, V)
    train_proj = step7_project_train(X_train, y_train, mean_face, U)

    # Chiếu test + nhận dạng
    pred, distances = step8_recognize(x_test, y_true, mean_face, U, train_proj, y_train)

    # Trực quan hóa
    x_centered_test = x_test - mean_face
    y_hat_test      = U.T @ x_centered_test
    step9_visualize(X_train, y_train, x_test, mean_face, U,
                    train_proj, y_hat_test, distances)

    print("\n" + "=" * 65)
    print("  Bước 5 hoàn tất.")
    print("=" * 65)
