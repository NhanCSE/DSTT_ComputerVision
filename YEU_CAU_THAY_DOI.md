# YÊU CẦU THAY ĐỔI: Chuyển ứng dụng "Khử nhiễu" thành "Làm mờ ảnh"

## 1. Mục tiêu

Thay đổi một trong các ứng dụng của dự án từ "Khử nhiễu ảnh" (Image Denoising) thành "Làm mờ ảnh" (Image Blurring). Kỹ thuật làm mờ ảnh này vẫn phải là một ứng dụng của phép chiếu vuông góc, cụ thể là tái tạo ảnh với số lượng eigenfaces (thành phần chính) thấp để loại bỏ các chi tiết tần số cao.

## 2. Các file cần thay đổi

- `README.md`
- `IMPLEMENTATION_PLAN.md`
- `extended_applications.py`
- `main_projection.py`

## 3. Hướng dẫn thay đổi chi tiết

### 3.1. `extended_applications.py` (Thay đổi chính)

Đây là file cần thay đổi nhiều nhất.

1.  **Cập nhật Docstring đầu file:**
    *   Thay đổi "Ứng dụng 2 — Khử nhiễu ảnh" thành "**Ứng dụng 2 — Làm mờ ảnh (Image Blurring)**".
    *   Cập nhật phần giải thích: Ý tưởng là các eigenface đầu tiên (eigenvalue lớn) nắm bắt các đặc trưng tần số thấp (cấu trúc tổng thể), trong khi các eigenface sau nắm bắt chi tiết tần số cao. Việc tái tạo ảnh chỉ với `k` eigenface đầu tiên hoạt động như một bộ lọc thông thấp (low-pass filter), loại bỏ chi tiết và gây ra hiệu ứng làm mờ.

2.  **Xóa code không cần thiết:**
    *   Xóa hàm `add_gaussian_noise`, vì chúng ta không còn làm việc với nhiễu nữa.

3.  **Viết lại "ỨNG DỤNG 2":**
    *   Đổi tên section comment từ `# ỨNG DỤNG 2 — Khử nhiễu ảnh` thành `# ỨNG DỤNG 2 — Làm mờ ảnh`.
    *   Xóa hoàn toàn hai hàm `plot_denoising_comparison` và `plot_denoising_psnr`.
    *   Tạo một hàm mới `plot_blurring_effect` thay thế. Hàm này có thể dựa trên `plot_reconstruction_comparison` nhưng với mục đích và tiêu đề khác.
        ```python
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
            # Nội dung hàm này sẽ tương tự plot_reconstruction_comparison:
            # 1. Vòng lặp qua k_values.
            # 2. Với mỗi k, gọi recognizer.reconstruct(x_sample, n_components=k).
            # 3. Tính MSE và PSNR so với ảnh gốc.
            # 4. Dùng matplotlib để vẽ ảnh gốc và các ảnh đã làm mờ tương ứng với từng k.
            # 5. Đặt tiêu đề chính là "Ứng dụng 2: Làm mờ ảnh bằng Phép chiếu vuông góc".
            # 6. Chú thích MSE/PSNR dưới mỗi ảnh.
            # 7. Lưu ảnh và trả về results.
        ```

4.  **Cập nhật `run_extended_applications`:**
    *   Trong phần `ỨNG DỤNG 2`, thay đổi tiêu đề in ra console thành "Làm mờ ảnh (Image Blurring)".
    *   Xóa logic chọn `k_denoise`. Thay vào đó, định nghĩa một danh sách `k_values_blur` với các giá trị `k` nhỏ để thấy rõ hiệu ứng mờ, ví dụ: `[1, 3, 7, 15, 30]`.
    *   Gọi hàm `plot_blurring_effect` mới thay cho hai hàm `plot_denoising_*`.
    *   Xóa hàm `plot_denoising_psnr` và lời gọi nó. Một biểu đồ so sánh trực quan là đủ cho ứng dụng làm mờ.
    *   Cập nhật bảng kết quả in ra console để hiển thị MSE/PSNR cho các mức độ mờ khác nhau (theo `k`).

### 3.2. `README.md`

1.  **Bảng ứng dụng (Section 1):**
    *   Thay đổi hàng thứ ba:
        *   **Ứng dụng:** `Làm mờ ảnh`
        *   **Mô tả:** `Làm mờ ảnh bằng cách tái tạo với số lượng eigenfaces thấp, hoạt động như một bộ lọc thông thấp.`

2.  **Mô tả module (Section 6):**
    *   Cập nhật mô tả cho `extended_applications.py`. Thay thế toàn bộ phần "Ứng dụng 2: Khử nhiễu ảnh" bằng mô tả cho "Ứng dụng 2: Làm mờ ảnh" và các hàm/file output mới (`plot_blurring_effect`, `app2_blurring.png`).

3.  **Kết quả kỳ vọng (Section 7):**
    *   Xóa bảng "Hiệu quả khử nhiễu".

4.  **Biểu đồ đầu ra (Section 8):**
    *   Thay đổi các file `app2_*`:
        *   `app2_denoising.png` -> `app2_blurring.png`
        *   Xóa `app2_psnr_gain.png`.

### 3.3. `IMPLEMENTATION_PLAN.md`

1.  **Bước 6 (Section 4):**
    *   Thay đổi tiêu đề "Ứng dụng 2" từ `Làm mờ / Khử nhiễu ảnh (Image Denoising via Low-rank Projection)` thành `Làm mờ ảnh (Image Blurring via Low-rank Projection)`.
    *   Cập nhật mô tả bên dưới để chỉ tập trung vào việc làm mờ bằng cách loại bỏ các thành phần tần số cao (eigenfaces cuối). Xóa các đề cập đến nhiễu Gaussian.

### 3.4. `main_projection.py`

1.  **Tóm tắt cuối (Section `TÓM TẮT KẾT QUẢ`):**
    *   Cập nhật danh sách các file biểu đồ đã lưu trong `outputs/` để phản ánh thay đổi từ `README.md` (thay `app2_denoising.png` và `app2_psnr_gain.png` bằng `app2_blurring.png`).