"""
Bước 6: Ứng dụng mở rộng — Tái tạo ảnh & Làm mờ ảnh bằng Phép chiếu vuông góc
================================================================================
Hai ứng dụng này minh họa sức mạnh của phép chiếu vuông góc NGOÀI bài toán
nhận dạng khuôn mặt, đáp ứng yêu cầu "ứng dụng trong nhiều lĩnh vực" của đề.

Cả hai đều dùng cùng một phép chiếu:
    x̂ = U_k · U_k^T · (x − x̄) + x̄

Ứng dụng 1 — Tái tạo ảnh (Image Reconstruction):
    Mục tiêu : nén ảnh bằng cách chỉ lưu k tọa độ Eigenface thay vì p pixel.
    Câu hỏi  : cần bao nhiêu Eigenfaces để tái tạo gần đúng ảnh gốc?
    Chỉ số   : MSE (càng nhỏ càng tốt) và PSNR (càng lớn càng tốt).

Ứng dụng 2 — Làm mờ ảnh (Image Blurring):
    Ý tưởng  : các eigenface đầu tiên (eigenvalue lớn) nắm bắt các đặc trưng
                tần số thấp (cấu trúc tổng thể của khuôn mặt), trong khi các
                eigenface sau nắm bắt chi tiết tần số cao (cạnh, nếp nhăn, mắt...).
                Việc tái tạo ảnh chỉ với k eigenface đầu tiên hoạt động như
                một bộ lọc thông thấp (low-pass filter): loại bỏ chi tiết
                tần số cao và gây ra hiệu ứng làm mờ. Mức độ mờ tăng khi k giảm.
    Chỉ số   : MSE và PSNR giữa ảnh gốc và ảnh đã làm mờ theo từng k.

Công thức:
    MSE  = (1/p) · Σ (x̂_i − x_i)²
    PSNR = 10 · log₁₀(255² / MSE)   [đơn vị: dB]

Tài liệu tham khảo:
    [1] Turk & Pentland (1991) — Eigenfaces for Recognition.
    [2] Jolliffe, I. T. (2002). "Principal Component Analysis." 2nd ed. Springer.
    [3] Gonzalez & Woods (2018). "Digital Image Processing." 4th ed. Pearson.
        (định nghĩa PSNR, MSE và bộ lọc thông thấp trong xử lý ảnh số)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào sys.path để import các module trong src/
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


# ==============================================================================
# METRICS — MSE và PSNR
# ==============================================================================

def mse(x_original: np.ndarray, x_reconstructed: np.ndarray) -> float:
    """
    Mean Squared Error giữa ảnh gốc và ảnh tái tạo (hoặc đã làm mờ).

    MSE = (1/p) · Σᵢ (x̂ᵢ − xᵢ)²

    MSE = 0  →  tái tạo hoàn hảo.
    MSE tăng khi k nhỏ (nén mạnh / mờ nhiều) — mất nhiều chi tiết tần số cao.
    """
    return float(np.mean((x_original - x_reconstructed) ** 2))


def psnr(x_original: np.ndarray, x_reconstructed: np.ndarray,
         max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio [dB].

    PSNR = 10 · log₁₀(MAX² / MSE)

    Quy tắc thực tế (ảnh grayscale 8-bit) [3]:
        PSNR > 40 dB  : chất lượng rất tốt (khó phân biệt với gốc)
        30–40 dB      : chất lượng tốt
        20–30 dB      : chất lượng chấp nhận được
        < 20 dB       : chất lượng kém (sai khác rõ ràng so với ảnh gốc)

    Nếu MSE = 0, trả về inf (tái tạo hoàn hảo).
    """
    err = mse(x_original, x_reconstructed)
    if err < 1e-12:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / err))


# ==============================================================================
# TIỆN ÍCH
# ==============================================================================

def _save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Lưu: {path}")
    plt.close(fig)


def _vec_to_img(vec: np.ndarray, h: int, w: int) -> np.ndarray:
    return vec.reshape(h, w)


# ==============================================================================
# ỨNG DỤNG 1 — Tái tạo ảnh
# ==============================================================================

def plot_reconstruction_comparison(
    recognizer,
    x_sample: np.ndarray,
    img_shape: tuple[int, int],
    k_values: list[int],
    save_path: str = "outputs/app1_reconstruction.png",
) -> dict:
    """
    So sánh trực quan ảnh gốc với ảnh tái tạo dùng k eigenfaces khác nhau.

    Công thức tái tạo:
        x̂ = U_k · U_k^T · (x − x̄) + x̄

    Giải thích hình học:
        - Chiếu x lên không gian span(u₁, ..., u_k)  →  tọa độ ŷ ∈ R^k
        - Chiếu ngược lại từ R^k → R^p               →  x̂ ∈ R^p
        - Sai số tái tạo = phần của (x − x̄) VUÔNG GÓC với span(u₁,...,u_k)
        - Khi k tăng, không gian con mở rộng, sai số giảm.

    Trả về:
        dict: {'k': k, 'mse': ..., 'psnr': ..., 'x_hat': ...} cho từng k
    """
    h, w    = img_shape
    results = {}

    # Tính tái tạo cho từng k
    for k in k_values:
        x_hat = recognizer.reconstruct(x_sample, n_components=k)
        results[k] = {
            "x_hat": x_hat,
            "mse"  : mse(x_sample, x_hat),
            "psnr" : psnr(x_sample, x_hat),
        }

    # ----- Vẽ biểu đồ -----
    n_cols = len(k_values) + 1   # +1 cho ảnh gốc
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.6, 3.8))

    # Ảnh gốc
    axes[0].imshow(_vec_to_img(x_sample, h, w), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh gốc\n(Original)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Ảnh tái tạo với từng k
    for col, k in enumerate(k_values, start=1):
        r      = results[k]
        x_hat  = r["x_hat"]
        axes[col].imshow(_vec_to_img(x_hat, h, w), cmap="gray", vmin=0, vmax=255)
        axes[col].set_title(
            f"k = {k} eigenfaces",
            fontsize=10,
        )
        axes[col].set_xlabel(
            f"MSE = {r['mse']:.1f}\nPSNR = {r['psnr']:.1f} dB",
            fontsize=8,
        )
        axes[col].axis("off")

    fig.suptitle(
        "Ứng dụng 1: Tái tạo ảnh bằng Phép chiếu vuông góc\n"
        r"$\hat{x} = U_k U_k^T (x - \bar{x}) + \bar{x}$",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return results


def plot_reconstruction_quality(
    recon_results: dict,
    save_path: str = "outputs/app1_quality_curve.png",
) -> None:
    """
    Vẽ đường cong MSE và PSNR theo số eigenfaces k.

    Cho thấy:
        - MSE giảm nhanh lúc đầu (các eigenfaces đầu chứa nhiều thông tin)
        - PSNR tăng tương ứng, tiếp cận giá trị "rất tốt" (>40 dB)
        - Diminishing returns: sau một ngưỡng, thêm eigenface ít cải thiện hơn
    """
    k_vals = sorted(recon_results.keys())
    mses   = [recon_results[k]["mse"]  for k in k_vals]
    psnrs  = [recon_results[k]["psnr"] for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # MSE
    ax1.plot(k_vals, mses, marker="o", color="#e74c3c", linewidth=2, markersize=6)
    ax1.fill_between(k_vals, mses, alpha=0.15, color="#e74c3c")
    ax1.set_xlabel("Số Eigenfaces (k)", fontsize=11)
    ax1.set_ylabel("MSE (↓ tốt hơn)", fontsize=11)
    ax1.set_title("Sai số tái tạo (MSE)", fontsize=12)
    ax1.grid(True, alpha=0.35)
    ax1.set_xticks(k_vals)

    # PSNR + ngưỡng chất lượng
    ax2.plot(k_vals, psnrs, marker="o", color="#27ae60", linewidth=2, markersize=6)
    ax2.fill_between(k_vals, psnrs, alpha=0.15, color="#27ae60")
    ax2.axhline(40, color="gray", linestyle="--", linewidth=1,
                label="Ngưỡng 40 dB (rất tốt)")
    ax2.axhline(30, color="orange", linestyle="--", linewidth=1,
                label="Ngưỡng 30 dB (tốt)")
    ax2.set_xlabel("Số Eigenfaces (k)", fontsize=11)
    ax2.set_ylabel("PSNR [dB] (↑ tốt hơn)", fontsize=11)
    ax2.set_title("Chất lượng tái tạo (PSNR)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    ax2.set_xticks(k_vals)

    fig.suptitle("Chất lượng tái tạo ảnh theo số Eigenfaces",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ==============================================================================
# ỨNG DỤNG 2 — Làm mờ ảnh
# ==============================================================================

def plot_blurring_effect(
    recognizer,
    x_sample: np.ndarray,
    img_shape: tuple[int, int],
    k_values: list[int],
    save_path: str = "outputs/app2_blurring.png",
) -> dict:
    """
    Minh họa hiệu ứng làm mờ bằng cách tái tạo ảnh với số lượng eigenfaces (k) thấp.

    Ý tưởng:
        - Tái tạo ảnh với k thấp loại bỏ các thành phần tần số cao (chi tiết nhỏ),
          chỉ giữ lại các thành phần tần số thấp (cấu trúc chính).
        - Đây là một dạng của bộ lọc thông thấp (low-pass filter), gây ra hiệu ứng làm mờ.
        - Mức độ mờ tăng khi k giảm.

    Công thức tái tạo:
        x̂ = U_k · U_k^T · (x − x̄) + x̄

    Trả về:
        dict: {'k': k, 'mse': ..., 'psnr': ...} cho từng k
    """
    h, w    = img_shape
    results = {}

    # Tính ảnh đã làm mờ cho từng k
    for k in k_values:
        x_blur = recognizer.reconstruct(x_sample, n_components=k)
        results[k] = {
            "x_blur": x_blur,
            "mse"   : mse(x_sample, x_blur),
            "psnr"  : psnr(x_sample, x_blur),
        }

    # ----- Vẽ biểu đồ -----
    n_cols = len(k_values) + 1   # +1 cho ảnh gốc
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.6, 3.8))

    # Ảnh gốc
    axes[0].imshow(_vec_to_img(x_sample, h, w), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh gốc\n(Original)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Ảnh đã làm mờ với từng k (k nhỏ → mờ nhiều, k lớn → mờ ít)
    for col, k in enumerate(k_values, start=1):
        r      = results[k]
        x_blur = r["x_blur"]
        axes[col].imshow(_vec_to_img(x_blur, h, w), cmap="gray", vmin=0, vmax=255)
        axes[col].set_title(
            f"k = {k} eigenfaces",
            fontsize=10,
        )
        axes[col].set_xlabel(
            f"MSE = {r['mse']:.1f}\nPSNR = {r['psnr']:.1f} dB",
            fontsize=8,
        )
        axes[col].axis("off")

    fig.suptitle(
        "Ứng dụng 2: Làm mờ ảnh bằng Phép chiếu vuông góc\n"
        r"$\hat{x} = U_k U_k^T (x - \bar{x}) + \bar{x}$  —  k nhỏ ⇒ chỉ giữ tần số thấp ⇒ ảnh mờ hơn",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return results


# ==============================================================================
# HÀM TỔNG HỢP — Chạy cả hai ứng dụng
# ==============================================================================

def run_extended_applications(
    recognizer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    img_shape: tuple[int, int],
    output_dir: str = "outputs",
) -> None:
    """
    Chạy toàn bộ Bước 6: sinh tất cả biểu đồ cho 2 ứng dụng mở rộng.

    Tự chọn một ảnh test đại diện (lấy ảnh đầu tiên) để minh họa.
    """
    # Lấy một ảnh test để làm ví dụ (người đầu tiên trong tập test)
    sample_idx = 0
    x_sample   = X_test[sample_idx]
    print(f"\n  Ảnh minh họa: X_test[{sample_idx}]  (Person {y_test[sample_idx]})")

    # ------------------------------------------------------------------
    # ỨNG DỤNG 1 — Tái tạo ảnh
    # ------------------------------------------------------------------
    print("\n── ỨNG DỤNG 1: Tái tạo ảnh (Image Reconstruction) ──")
    k_values_recon = [1, 5, 20, 50, 100, 150]
    k_values_recon = [k for k in k_values_recon
                      if k <= recognizer.eigenfaces_.shape[1]]

    recon_results = plot_reconstruction_comparison(
        recognizer,
        x_sample,
        img_shape,
        k_values=k_values_recon,
        save_path=os.path.join(output_dir, "app1_reconstruction.png"),
    )

    plot_reconstruction_quality(
        recon_results,
        save_path=os.path.join(output_dir, "app1_quality_curve.png"),
    )

    # In bảng kết quả ra console
    print(f"\n  {'k':>6}  {'MSE':>10}  {'PSNR (dB)':>12}")
    print("  " + "-" * 34)
    for k in sorted(recon_results):
        r = recon_results[k]
        print(f"  {k:>6}  {r['mse']:>10.2f}  {r['psnr']:>12.2f}")

    # ------------------------------------------------------------------
    # ỨNG DỤNG 2 — Làm mờ ảnh
    # ------------------------------------------------------------------
    print("\n── ỨNG DỤNG 2: Làm mờ ảnh (Image Blurring) ──")

    # Dùng k nhỏ để thấy rõ hiệu ứng mờ (k càng nhỏ → ảnh càng mờ)
    k_values_blur = [1, 3, 7, 15, 30]
    k_values_blur = [k for k in k_values_blur
                     if k <= recognizer.eigenfaces_.shape[1]]
    print(f"  Tái tạo ảnh với k nhỏ → bộ lọc thông thấp → hiệu ứng làm mờ")
    print(f"  k thử nghiệm: {k_values_blur}\n")

    blur_results = plot_blurring_effect(
        recognizer,
        x_sample,
        img_shape,
        k_values=k_values_blur,
        save_path=os.path.join(output_dir, "app2_blurring.png"),
    )

    # In bảng kết quả ra console: MSE/PSNR cho các mức độ mờ khác nhau
    print(f"\n  {'k':>6}  {'MSE':>10}  {'PSNR (dB)':>12}  {'Mức mờ':>10}")
    print("  " + "-" * 46)
    for k in sorted(blur_results):
        r     = blur_results[k]
        # Quy ước: k càng nhỏ thì ảnh càng mờ
        level = "rất mờ" if k <= 3 else ("mờ" if k <= 15 else "ít mờ")
        print(f"  {k:>6}  {r['mse']:>10.2f}  {r['psnr']:>12.2f}  {level:>10}")

    print(f"\n  Tất cả biểu đồ Bước 6 đã lưu vào: {output_dir}/")


# ==============================================================================
# CHẠY THỬ
# ==============================================================================

if __name__ == "__main__":
    from src.dataloader import load_orl_dataset, download_orl_dataset, IMG_HEIGHT, IMG_WIDTH
    from src.recognizer import OrthogonalFaceRecognizer

    data_dir = os.path.join(_ROOT, "data", "orl_faces")
    if not os.path.isdir(data_dir):
        download_orl_dataset(save_dir=os.path.join(_ROOT, "data"))

    print("Đang đọc dataset ORL Faces...")
    X_train, y_train, X_test, y_test = load_orl_dataset(data_dir=data_dir)

    print("Đang huấn luyện OrthogonalFaceRecognizer (k=150)...")
    # Dùng k lớn để có đủ eigenfaces cho mọi thí nghiệm tái tạo
    recognizer = OrthogonalFaceRecognizer(n_components=150).fit(X_train, y_train)
    recognizer.print_summary()

    run_extended_applications(
        recognizer  = recognizer,
        X_test      = X_test,
        y_test      = y_test,
        img_shape   = (IMG_HEIGHT, IMG_WIDTH),
        output_dir  = os.path.join(_ROOT, "outputs"),
    )

    print("\nBước 6 hoàn tất.")
    print("Toàn bộ dự án đã sẵn sàng. Xem thư mục outputs/ để lấy biểu đồ cho báo cáo.")
