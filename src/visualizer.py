"""
Bước 4: Trực quan hóa (Visualization)
======================================
Sinh các biểu đồ cho báo cáo và slide, lưu vào thư mục outputs/.

Danh sách hình:
  Fig 1 — mean_face.png        : Khuôn mặt trung bình
  Fig 2 — eigenfaces.png       : Lưới các khuôn mặt đặc trưng (Eigenfaces)
  Fig 3 — recognition.png      : Ví dụ nhận dạng đúng và nhận dạng sai
  Fig 4 — accuracy_vs_k.png    : Accuracy của Eigenfaces theo số k
  Fig 5 — variance_ratio.png   : Tỉ lệ phương sai giải thích theo eigenface

Tài liệu tham khảo:
  [1] Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment."
      Computing in Science & Engineering, 9(3), 90–95.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import IMG_HEIGHT, IMG_WIDTH


# ==============================================================================
# TIỆN ÍCH NỘI BỘ
# ==============================================================================

def _vec_to_img(vec: np.ndarray) -> np.ndarray:
    """Duỗi ngược: chuyển vector 1D (p,) → ảnh 2D (IMG_HEIGHT, IMG_WIDTH)."""
    return vec.reshape(IMG_HEIGHT, IMG_WIDTH)


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa giá trị về [0, 1] để hiển thị.
    Cần thiết cho eigenfaces vì chúng có thể có giá trị ÂM
    (eigenface là vector toán học, không phải ảnh thật).

    Công thức min-max scaling:
        out = (img - min) / (max - min)
    → out có giá trị nhỏ nhất = 0, lớn nhất = 1.
    """
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-10:          # ảnh hằng số (tránh chia cho 0)
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _save(fig: plt.Figure, path: str) -> None:
    """Lưu figure ra file với DPI cao (phù hợp in báo cáo)."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Đã lưu: {path}")
    plt.close(fig)


# ==============================================================================
# HÌNH 1 — Khuôn mặt trung bình (Mean Face)
# ==============================================================================

def plot_mean_face(
    recognizer,
    save_path: str = "outputs/mean_face.png",
) -> None:
    """
    Hiển thị khuôn mặt trung bình x̄ = (1/N) Σ xi.

    x̄ là "nguyên mẫu" của tập ảnh huấn luyện. Mọi ảnh đặc trưng (eigenface)
    đều là hướng sai lệch so với khuôn mặt này.
    """
    mean_img = _vec_to_img(recognizer.mean_face_)  # shape: (112, 92)

    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    ax.imshow(mean_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Khuôn mặt trung bình\n(Mean Face  $\\bar{x}$)", fontsize=13)
    ax.axis("off")

    fig.suptitle(
        f"Tính từ {len(recognizer.train_labels_)} ảnh huấn luyện",
        fontsize=9, color="gray", y=0.02,
    )
    _save(fig, save_path)


# ==============================================================================
# HÌNH 2 — Lưới Eigenfaces (Khuôn mặt đặc trưng)
# ==============================================================================

def plot_eigenfaces(
    recognizer,
    n_faces: int = 20,
    save_path: str = "outputs/eigenfaces.png",
) -> None:
    """
    Hiển thị lưới n_faces eigenfaces đầu tiên (theo thứ tự eigenvalue giảm dần).

    Eigenface thứ i (u_i) là hướng quan trọng thứ i trong không gian ảnh —
    hướng mà dữ liệu biến thiên nhiều nhất sau khi đã loại bỏ u_1, ..., u_{i-1}.

    Lưu ý: eigenfaces có giá trị âm → cần normalize về [0,1] để hiển thị.
    """
    k_avail = recognizer.eigenfaces_.shape[1]
    n       = min(n_faces, k_avail)

    # Bố cục lưới: cố gắng gần hình vuông
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.4))
    axes = axes.flatten()

    evr = recognizer.explained_variance_ratio()  # tỉ lệ phương sai từng eigenface

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            # Lấy eigenface thứ i, reshape về 2D, normalize để hiển thị
            ef_img = _normalize_for_display(
                _vec_to_img(recognizer.eigenfaces_[:, i])
            )
            ax.imshow(ef_img, cmap="gray")
            ax.set_title(
                f"#{i+1}\n{evr[i]*100:.1f}%",
                fontsize=8, pad=2,
            )
        ax.axis("off")

    fig.suptitle(
        f"Top-{n} Eigenfaces (Khuôn mặt đặc trưng)\n"
        f"Tổng phương sai giải thích: {evr[:n].sum()*100:.1f}%",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, save_path)


# ==============================================================================
# HÌNH 3 — Ví dụ nhận dạng đúng và sai
# ==============================================================================

def plot_recognition_examples(
    recognizer,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = "outputs/recognition.png",
) -> None:
    """
    Minh họa một ví dụ nhận dạng ĐÚNG và một ví dụ nhận dạng SAI.

    Mỗi ví dụ gồm 3 ảnh:
      [Ảnh truy vấn]  →  [Ảnh khớp gần nhất]  |  [Ảnh đúng của người đó]

    Viền xanh lá = đúng, viền đỏ = sai.
    """
    y_pred = recognizer.predict(X_test)

    correct_mask = y_pred == y_test
    wrong_mask   = ~correct_mask

    # Lấy chỉ số ví dụ đúng và sai đầu tiên tìm được
    correct_indices = np.where(correct_mask)[0]
    wrong_indices   = np.where(wrong_mask)[0]

    if len(correct_indices) == 0:
        print("  [WARN] Không có ví dụ nhận dạng đúng nào!")
        correct_idx = None
    else:
        correct_idx = correct_indices[0]

    if len(wrong_indices) == 0:
        print("  [WARN] Không có ví dụ nhận dạng sai nào (accuracy = 100%)!")
        wrong_idx = None
    else:
        wrong_idx = wrong_indices[0]

    # Hàm tìm ảnh train gần nhất (nearest neighbor trong eigenface space)
    def _find_nearest_train(test_vec: np.ndarray) -> int:
        test_proj  = recognizer.project(test_vec)
        diffs      = recognizer.train_projections_ - test_proj
        distances  = np.sqrt(np.sum(diffs ** 2, axis=1))
        return int(np.argmin(distances))

    # Hàm lấy một ảnh đại diện của person_id trong tập train
    def _get_ref_image(person_id: int) -> np.ndarray:
        mask = y_train == person_id
        return X_train[mask][0]

    # Xây dựng danh sách các hàng cần vẽ
    rows = []
    if correct_idx is not None:
        nn_idx    = _find_nearest_train(X_test[correct_idx])
        rows.append({
            "label"     : f"ĐÚNG — Person {y_test[correct_idx]}",
            "color"     : "#2ecc71",
            "query"     : X_test[correct_idx],
            "matched"   : X_train[nn_idx],
            "actual"    : None,            # đúng → không cần ảnh "đúng" riêng
            "pred_id"   : y_pred[correct_idx],
            "true_id"   : y_test[correct_idx],
        })
    if wrong_idx is not None:
        nn_idx    = _find_nearest_train(X_test[wrong_idx])
        rows.append({
            "label"     : f"SAI — Đoán: Person {y_pred[wrong_idx]}  |  Thực: Person {y_test[wrong_idx]}",
            "color"     : "#e74c3c",
            "query"     : X_test[wrong_idx],
            "matched"   : X_train[nn_idx],
            "actual"    : _get_ref_image(y_test[wrong_idx]),
            "pred_id"   : y_pred[wrong_idx],
            "true_id"   : y_test[wrong_idx],
        })

    n_rows  = len(rows)
    n_cols  = 3          # query | matched | actual (chỉ khi sai)
    fig_h   = n_rows * 3.4 + 1.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, fig_h))
    # Đảm bảo axes luôn là 2D dù n_rows == 1
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    fig.suptitle("Ví dụ nhận dạng khuôn mặt", fontsize=14, fontweight="bold")

    col_titles = [
        "Ảnh truy vấn\n(Test image)",
        "Ảnh khớp gần nhất\n(Nearest neighbor)",
        "Ảnh đúng\n(Ground truth)",
    ]

    for r, row in enumerate(rows):
        images    = [row["query"], row["matched"], row["actual"]]
        subtitles = [
            f"Person {row['true_id']}",
            f"Khớp với Person {row['pred_id']}",
            f"Person {row['true_id']} (thực)",
        ]
        for c in range(n_cols):
            ax = axes[r, c]

            if images[c] is not None:
                ax.imshow(_vec_to_img(images[c]), cmap="gray", vmin=0, vmax=255)
            else:
                # Cột "actual" khi kết quả đúng → hiển thị ô trống có dấu tích
                ax.text(0.5, 0.5, "✓ Đúng", transform=ax.transAxes,
                        ha="center", va="center", fontsize=16,
                        color=row["color"], fontweight="bold")
                ax.set_facecolor("#f0f0f0")

            # Viền màu cho cột ảnh truy vấn (xanh = đúng, đỏ = sai)
            if c == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor(row["color"])
                    spine.set_linewidth(3)

            # Tiêu đề cột chỉ hiển thị ở hàng đầu
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9, pad=4)

            ax.set_xlabel(subtitles[c], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Thêm nhãn kết quả bên trái mỗi hàng dùng fig.text()
        # y_pos: ước tính vị trí giữa hàng r trong hệ tọa độ figure [0,1]
        y_pos = 1.0 - (r + 0.5) / n_rows
        fig.text(
            0.01, y_pos, row["label"],
            ha="left", va="center",
            fontsize=8, color=row["color"], fontweight="bold",
            rotation=90,
        )

    fig.tight_layout(rect=[0.04, 0, 1, 0.97])  # chừa lề trái cho nhãn
    _save(fig, save_path)


# ==============================================================================
# HÌNH 4 — Accuracy vs k (từ kết quả Bước 3)
# ==============================================================================

def plot_accuracy_vs_k(
    k_results: dict,
    baseline_accuracy: float | None = None,
    save_path: str = "outputs/accuracy_vs_k.png",
) -> None:
    """
    Vẽ đường cong Accuracy của Eigenfaces theo số lượng eigenfaces k.

    Cho thấy:
      - Khi k quá nhỏ: thiếu thông tin → accuracy thấp.
      - k tối ưu: "điểm ngọt" (sweet spot) cân bằng nén và thông tin.
      - Khi k quá lớn: bắt đầu học nhiễu → accuracy có thể giảm nhẹ.
    """
    k_vals    = k_results["k_values"]
    accs      = [a * 100 for a in k_results["accuracies"]]
    best_k    = k_results["best_k"]
    best_acc  = k_results["best_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Đường accuracy của Eigenfaces
    ax.plot(k_vals, accs, marker="o", color="#3498db", linewidth=2,
            markersize=6, label="Eigenfaces (Phép chiếu vuông góc)")

    # Đánh dấu điểm k tốt nhất
    ax.annotate(
        f"k tốt nhất = {best_k}\nAccuracy = {best_acc:.1f}%",
        xy=(best_k, best_acc),
        xytext=(best_k + max(k_vals) * 0.06, best_acc - 5),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )
    ax.scatter([best_k], [best_acc], color="#e74c3c", s=80, zorder=5)

    # Đường baseline
    if baseline_accuracy is not None:
        ax.axhline(
            y=baseline_accuracy * 100,
            color="#e67e22", linewidth=1.8, linestyle="--",
            label=f"Baseline Pixel-KNN ({baseline_accuracy*100:.1f}%)",
        )

    ax.set_xlabel("Số lượng Eigenfaces giữ lại (k)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Độ chính xác nhận dạng theo số Eigenfaces", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(bottom=max(0, min(accs) - 10), top=102)
    ax.set_xticks(k_vals)

    fig.tight_layout()
    _save(fig, save_path)


# ==============================================================================
# HÌNH 5 — Tỉ lệ phương sai giải thích (Explained Variance)
# ==============================================================================

def plot_explained_variance(
    recognizer,
    save_path: str = "outputs/variance_ratio.png",
) -> None:
    """
    Biểu đồ kép:
      - Cột xanh: phương sai giải thích bởi từng eigenface (individual).
      - Đường cam: phương sai tích lũy (cumulative).
      - Đường ngang đứt: ngưỡng 95%.

    Cho thấy rõ eigenface đầu tiên chiếm phần lớn variance, các eigenface
    sau đóng góp ít hơn nhưng vẫn cần thiết để đạt ngưỡng mong muốn.
    """
    evr    = recognizer.explained_variance_ratio() * 100    # %
    cumvar = recognizer.cumulative_explained_variance() * 100
    k_95   = recognizer.n_components_for_variance(0.95)
    indices = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()   # trục y thứ hai (bên phải) cho cumulative

    # Cột: phương sai từng eigenface
    ax1.bar(indices, evr, color="#3498db", alpha=0.55, label="Từng eigenface")
    ax1.set_xlabel("Eigenface thứ i", fontsize=11)
    ax1.set_ylabel("Phương sai giải thích (%)", color="#3498db", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#3498db")

    # Đường: phương sai tích lũy
    ax2.plot(indices, cumvar, color="#e67e22", linewidth=2.2,
             marker="", label="Tích lũy")
    ax2.set_ylabel("Phương sai tích lũy (%)", color="#e67e22", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax2.set_ylim(0, 105)

    # Ngưỡng 95%
    ax2.axhline(y=95, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvline(x=k_95, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(k_95 + 1, 60, f"k={k_95}\nđạt 95%",
             fontsize=8, color="red",
             bbox=dict(boxstyle="round,pad=0.2", fc="mistyrose", ec="red", alpha=0.7))

    ax1.set_title("Phương sai giải thích theo số Eigenfaces", fontsize=13)

    # Gộp legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)


# ==============================================================================
# HÀM TỔNG HỢP — Sinh tất cả biểu đồ cùng lúc
# ==============================================================================

def plot_all(
    recognizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    k_results: dict | None = None,
    baseline_accuracy: float | None = None,
    output_dir: str = "outputs",
) -> None:
    """
    Sinh toàn bộ 5 biểu đồ và lưu vào output_dir/.

    Tham số:
        recognizer        : OrthogonalFaceRecognizer đã fit().
        X_train, y_train  : tập huấn luyện.
        X_test, y_test    : tập kiểm tra.
        k_results         : kết quả từ compare_k_values() (Bước 3), dùng cho Fig 4.
        baseline_accuracy : accuracy của Baseline (Bước 3), dùng cho Fig 4.
        output_dir        : thư mục lưu ảnh.
    """
    print("\n== Đang sinh các biểu đồ ==")

    plot_mean_face(
        recognizer,
        save_path=os.path.join(output_dir, "mean_face.png"),
    )

    plot_eigenfaces(
        recognizer,
        n_faces=20,
        save_path=os.path.join(output_dir, "eigenfaces.png"),
    )

    plot_recognition_examples(
        recognizer,
        X_test, y_test, X_train, y_train,
        save_path=os.path.join(output_dir, "recognition.png"),
    )

    if k_results is not None:
        plot_accuracy_vs_k(
            k_results,
            baseline_accuracy=baseline_accuracy,
            save_path=os.path.join(output_dir, "accuracy_vs_k.png"),
        )
    else:
        print("  [SKIP] accuracy_vs_k.png — cần truyền k_results từ Bước 3.")

    plot_explained_variance(
        recognizer,
        save_path=os.path.join(output_dir, "variance_ratio.png"),
    )

    print(f"\nTất cả biểu đồ đã lưu vào thư mục: {output_dir}/\n")


# ==============================================================================
# CHẠY THỬ
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from src.dataloader  import load_orl_dataset, download_orl_dataset
    from src.recognizer  import OrthogonalFaceRecognizer
    from src.evaluator   import compare_k_values, accuracy

    data_dir = os.path.join(project_root, "data", "orl_faces")
    if not os.path.isdir(data_dir):
        download_orl_dataset(save_dir=os.path.join(project_root, "data"))

    print("Đang đọc dataset...")
    X_train, y_train, X_test, y_test = load_orl_dataset(data_dir=data_dir)

    print("Đang huấn luyện mô hình (k=50)...")
    rec = OrthogonalFaceRecognizer(n_components=50).fit(X_train, y_train)
    rec.print_summary()

    print("Đang tìm k tối ưu...")
    k_results = compare_k_values(X_train, y_train, X_test, y_test)

    from src.baseline import PixelKNNRecognizer
    baseline_acc = accuracy(y_test, PixelKNNRecognizer(k=1).fit(X_train, y_train).predict(X_test))

    # Huấn luyện lại với k tốt nhất để vẽ biểu đồ
    best_k = k_results["best_k"]
    rec    = OrthogonalFaceRecognizer(n_components=best_k).fit(X_train, y_train)

    plot_all(
        recognizer        = rec,
        X_train           = X_train,
        y_train           = y_train,
        X_test            = X_test,
        y_test            = y_test,
        k_results         = k_results,
        baseline_accuracy = baseline_acc,
        output_dir        = os.path.join(project_root, "outputs"),
    )

    print("Bước 4 hoàn tất.")
