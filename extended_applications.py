"""
Bước 6: Ứng dụng mở rộng — Tái tạo ảnh & Khử nhiễu bằng Phép chiếu vuông góc
================================================================================
Hai ứng dụng này minh họa sức mạnh của phép chiếu vuông góc NGOÀI bài toán
nhận dạng khuôn mặt, đáp ứng yêu cầu "ứng dụng trong nhiều lĩnh vực" của đề.

Cả hai đều dùng cùng một phép chiếu:
    x̂ = U_k · U_k^T · (x − x̄) + x̄

Ứng dụng 1 — Tái tạo ảnh (Image Reconstruction):
    Mục tiêu : nén ảnh bằng cách chỉ lưu k tọa độ Eigenface thay vì p pixel.
    Câu hỏi  : cần bao nhiêu Eigenfaces để tái tạo gần đúng ảnh gốc?
    Chỉ số   : MSE (càng nhỏ càng tốt) và PSNR (càng lớn càng tốt).

Ứng dụng 2 — Khử nhiễu ảnh (Image Denoising via Low-rank Projection):
    Ý tưởng  : nhiễu trải đều mọi hướng trong R^p, còn tín hiệu (khuôn mặt)
                tập trung theo k hướng Eigenface đầu tiên.
                → Chiếu về không gian Eigenface giữ lại tín hiệu, loại bỏ nhiễu.
    Chỉ số   : PSNR tăng sau khi khử nhiễu so với ảnh bị nhiễu.

Công thức:
    MSE  = (1/p) · Σ (x̂_i − x_i)²
    PSNR = 10 · log₁₀(255² / MSE)   [đơn vị: dB]

Tài liệu tham khảo:
    [1] Turk & Pentland (1991) — Eigenfaces for Recognition.
    [2] Jolliffe, I. T. (2002). "Principal Component Analysis." 2nd ed. Springer.
    [3] Gonzalez & Woods (2018). "Digital Image Processing." 4th ed. Pearson.
        (định nghĩa PSNR, MSE trong xử lý ảnh số)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Thêm thư mục gốc vào sys.path để import các module trong src/
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


# ==============================================================================
# METRICS — MSE và PSNR
# ==============================================================================

def mse(x_original: np.ndarray, x_reconstructed: np.ndarray) -> float:
    """
    Mean Squared Error giữa ảnh gốc và ảnh tái tạo/khử nhiễu.

    MSE = (1/p) · Σᵢ (x̂ᵢ − xᵢ)²

    MSE = 0  →  tái tạo hoàn hảo.
    MSE tăng khi k nhỏ (nén mạnh) hoặc nhiễu lớn.
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
        < 20 dB       : chất lượng kém (nhiễu rõ ràng)

    Nếu MSE = 0, trả về inf (tái tạo hoàn hảo).
    """
    err = mse(x_original, x_reconstructed)
    if err < 1e-12:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / err))


# ==============================================================================
# TIỆN ÍCH
# ==============================================================================

def add_gaussian_noise(X: np.ndarray, sigma: float, rng_seed: int = 42) -> np.ndarray:
    """
    Thêm nhiễu Gaussian (White Gaussian Noise) vào ảnh.

    Mỗi pixel bị cộng thêm một giá trị ngẫu nhiên ε ~ N(0, σ²).
    Kết quả được clip về [0, 255] để giữ giá trị pixel hợp lệ.

    Tham số:
        X     : ảnh (vector hoặc ma trận) giá trị pixel trong [0, 255].
        sigma : độ lệch chuẩn của nhiễu (lớn hơn = nhiễu mạnh hơn).
        rng_seed : seed ngẫu nhiên để kết quả tái lập được.
    """
    rng   = np.random.default_rng(rng_seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return np.clip(X + noise, 0.0, 255.0)


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
# ỨNG DỤNG 2 — Khử nhiễu ảnh
# ==============================================================================

def plot_denoising_comparison(
    recognizer,
    x_sample: np.ndarray,
    img_shape: tuple[int, int],
    sigma_values: list[float],
    k_denoise: int,
    save_path: str = "outputs/app2_denoising.png",
) -> dict:
    """
    Minh họa khử nhiễu bằng phép chiếu vuông góc với nhiều mức nhiễu khác nhau.

    Tại sao chiếu xuống giúp khử nhiễu? [2][3]
        - Nhiễu Gaussian ~ N(0, σ²) phân tán đều theo MỌI hướng trong R^p.
        - Tín hiệu khuôn mặt tập trung theo k hướng Eigenface (phương sai lớn nhất).
        - Chiếu xuống không gian k chiều:
            + Giữ lại: phần tín hiệu dọc theo u₁,...,u_k
            + Loại bỏ: phần nhiễu theo (p−k) hướng vuông góc còn lại
        - Hiệu quả nhất khi k nhỏ hơn nhiều so với p (p = 10304, k ~ 50).

    Tham số:
        sigma_values : danh sách mức nhiễu σ cần thử.
        k_denoise    : số eigenfaces dùng để chiếu khử nhiễu.

    Trả về:
        dict: {sigma: {'noisy': ..., 'denoised': ..., 'psnr_noisy': ..., 'psnr_denoised': ...}}
    """
    h, w    = img_shape
    results = {}
    n_sigma = len(sigma_values)

    fig, axes = plt.subplots(n_sigma, 3, figsize=(9, n_sigma * 3.2))
    if n_sigma == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["Ảnh gốc\n(Original)", "Ảnh bị nhiễu\n(Noisy)", "Ảnh sau khử nhiễu\n(Denoised)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=8)

    for row, sigma in enumerate(sigma_values):
        # Thêm nhiễu
        x_noisy    = add_gaussian_noise(x_sample, sigma=sigma, rng_seed=42 + row)
        # Khử nhiễu bằng phép chiếu
        x_denoised = recognizer.reconstruct(x_noisy, n_components=k_denoise)

        psnr_noisy    = psnr(x_sample, x_noisy)
        psnr_denoised = psnr(x_sample, x_denoised)
        gain          = psnr_denoised - psnr_noisy

        results[sigma] = {
            "x_noisy"      : x_noisy,
            "x_denoised"   : x_denoised,
            "psnr_noisy"   : psnr_noisy,
            "psnr_denoised": psnr_denoised,
            "psnr_gain"    : gain,
        }

        imgs     = [x_sample, x_noisy, x_denoised]
        captions = [
            "—",
            f"PSNR = {psnr_noisy:.1f} dB",
            f"PSNR = {psnr_denoised:.1f} dB  (+{gain:.1f} dB)",
        ]
        border_colors = ["#2c3e50", "#e74c3c", "#27ae60"]

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(_vec_to_img(imgs[col], h, w), cmap="gray", vmin=0, vmax=255)
            ax.set_xlabel(captions[col], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_colors[col])
                spine.set_linewidth(2)

        # Nhãn mức nhiễu bên trái
        axes[row, 0].set_ylabel(f"σ = {sigma:.0f}", fontsize=11,
                                fontweight="bold", rotation=90, labelpad=8)

    fig.suptitle(
        f"Ứng dụng 2: Khử nhiễu bằng Phép chiếu vuông góc  (k = {k_denoise} Eigenfaces)\n"
        "Chiếu về không gian k chiều loại bỏ nhiễu theo (p−k) hướng vuông góc còn lại",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return results


def plot_denoising_psnr(
    denoise_results: dict,
    k_denoise: int,
    save_path: str = "outputs/app2_psnr_gain.png",
) -> None:
    """
    Biểu đồ so sánh PSNR trước và sau khi khử nhiễu theo từng mức nhiễu σ.
    """
    sigmas         = sorted(denoise_results.keys())
    psnrs_noisy    = [denoise_results[s]["psnr_noisy"]    for s in sigmas]
    psnrs_denoised = [denoise_results[s]["psnr_denoised"] for s in sigmas]
    gains          = [denoise_results[s]["psnr_gain"]     for s in sigmas]

    x      = np.arange(len(sigmas))
    width  = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # PSNR trước và sau
    bars1 = ax1.bar(x - width / 2, psnrs_noisy,    width, label="Trước khử nhiễu",
                    color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, psnrs_denoised, width, label="Sau khử nhiễu",
                    color="#27ae60", alpha=0.85)

    ax1.axhline(30, color="orange", linestyle="--", linewidth=1, label="30 dB (tốt)")
    ax1.axhline(40, color="gray",   linestyle="--", linewidth=1, label="40 dB (rất tốt)")

    ax1.set_xlabel("Mức nhiễu Gaussian (σ)", fontsize=11)
    ax1.set_ylabel("PSNR [dB]", fontsize=11)
    ax1.set_title("PSNR trước và sau khử nhiễu", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"σ={s:.0f}" for s in sigmas])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Gán số lên cột
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # Mức tăng PSNR
    ax2.bar(x, gains, color="#3498db", alpha=0.85)
    for i, g in enumerate(gains):
        ax2.text(i, g + 0.1, f"+{g:.1f} dB", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_xlabel("Mức nhiễu Gaussian (σ)", fontsize=11)
    ax2.set_ylabel("PSNR tăng thêm [dB]", fontsize=11)
    ax2.set_title(f"Mức tăng PSNR nhờ khử nhiễu (k = {k_denoise})", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"σ={s:.0f}" for s in sigmas])
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Hiệu quả khử nhiễu bằng Phép chiếu Eigenface", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, save_path)


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
    # ỨNG DỤNG 2 — Khử nhiễu
    # ------------------------------------------------------------------
    print("\n── ỨNG DỤNG 2: Khử nhiễu ảnh (Image Denoising) ──")

    # Chọn k khử nhiễu: dùng k giải thích ~80% variance (không quá nhiều để loại nhiễu tốt)
    k_denoise = recognizer.n_components_for_variance(0.80)
    k_denoise = min(k_denoise, recognizer.eigenfaces_.shape[1])
    print(f"  Dùng k = {k_denoise} eigenfaces để khử nhiễu "
          f"(giải thích ≥80% variance)\n")

    sigma_values  = [10.0, 25.0, 50.0]

    denoise_results = plot_denoising_comparison(
        recognizer,
        x_sample,
        img_shape,
        sigma_values=sigma_values,
        k_denoise=k_denoise,
        save_path=os.path.join(output_dir, "app2_denoising.png"),
    )

    plot_denoising_psnr(
        denoise_results,
        k_denoise=k_denoise,
        save_path=os.path.join(output_dir, "app2_psnr_gain.png"),
    )

    # In bảng kết quả ra console
    print(f"\n  {'σ':>6}  {'PSNR (nhiễu)':>14}  {'PSNR (khử)':>12}  {'Tăng':>8}")
    print("  " + "-" * 48)
    for sigma in sorted(denoise_results):
        r = denoise_results[sigma]
        print(f"  {sigma:>6.0f}  {r['psnr_noisy']:>14.1f}  "
              f"{r['psnr_denoised']:>12.1f}  {r['psnr_gain']:>+8.1f} dB")

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
