# Nhận dạng Đối tượng bằng Phép chiếu Vuông góc

**Bài tập lớn — Môn Đại số Tuyến tính**
**Đề tài:** Nghiên cứu tính chất và ứng dụng của phép chiếu vuông góc trong thị giác máy tính

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở toán học](#2-cơ-sở-toán-học)
3. [Cấu trúc dự án](#3-cấu-trúc-dự-án)
4. [Cài đặt](#4-cài-đặt)
5. [Hướng dẫn chạy](#5-hướng-dẫn-chạy)
6. [Mô tả chi tiết từng module](#6-mô-tả-chi-tiết-từng-module)
7. [Kết quả kỳ vọng](#7-kết-quả-kỳ-vọng)
8. [Biểu đồ đầu ra](#8-biểu-đồ-đầu-ra)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

Dự án này hiện thực thuật toán **Eigenfaces** — ứng dụng tiêu biểu của phép chiếu vuông góc trong thị giác máy tính — để giải quyết ba bài toán:

| Ứng dụng | Mô tả |
|---|---|
| **Nhận dạng khuôn mặt** | Nhận dạng danh tính từ ảnh mới bằng cách so sánh trong không gian eigenface |
| **Tái tạo ảnh** | Nén và tái tạo ảnh qua phép chiếu lên không gian con k chiều |
| **Làm mờ ảnh** | Làm mờ ảnh bằng cách tái tạo với số lượng eigenfaces thấp, hoạt động như một bộ lọc thông thấp. |

### Ràng buộc kỹ thuật

- **Không dùng hộp đen:** Toàn bộ đại số tuyến tính (hiệp phương sai, trị riêng, vector riêng, phép chiếu) được tự viết bằng `numpy` thuần — không dùng `sklearn.decomposition.PCA` hay bất kỳ hàm ML có sẵn nào.
- **Tính minh bạch:** Mọi khối code toán học đều có comment giải thích từng bước và trích dẫn tài liệu.

### Dataset

**AT&T Database of Faces** (ORL Faces):
- 40 người × 10 ảnh/người = **400 ảnh** tổng cộng
- Kích thước mỗi ảnh: **92 × 112 pixel**, grayscale
- Nguồn: AT&T Laboratories Cambridge

---

## 2. Cơ sở toán học

### 2.1 Vector hóa ảnh

Mỗi ảnh $m \times n$ pixel được duỗi thành vector $p = m \cdot n$ chiều:

$$\mathbf{x}_i \in \mathbb{R}^p, \quad p = 92 \times 112 = 10304$$

Tập $N$ ảnh huấn luyện tạo thành ma trận:

$$X = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N]^T \in \mathbb{R}^{N \times p}$$

### 2.2 Khuôn mặt trung bình

$$\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$

### 2.3 Trung tâm hóa dữ liệu

$$\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}, \quad \Phi \in \mathbb{R}^{N \times p}$$

### 2.4 Ma trận hiệp phương sai và thủ thuật tính nhanh

Ma trận hiệp phương sai thực sự:

$$C = \frac{1}{N} \Phi^T \Phi \in \mathbb{R}^{p \times p}$$

Với $p = 10304$, ma trận $C$ quá lớn để phân rã trực tiếp. **Thủ thuật Turk & Pentland (1991):** tính ma trận thay thế nhỏ hơn:

$$L = \frac{1}{N} \Phi \Phi^T \in \mathbb{R}^{N \times N}$$

**Chứng minh tính đúng đắn:**  
Nếu $L\mathbf{v} = \lambda\mathbf{v}$, nhân trái bởi $\Phi^T$:

$$\frac{1}{N}\Phi^T\Phi\underbrace{(\Phi^T\mathbf{v})}_{\mathbf{u}} = \lambda(\Phi^T\mathbf{v}) \implies C\mathbf{u} = \lambda\mathbf{u}$$

Vậy $\mathbf{u} = \Phi^T\mathbf{v}$ là eigenvector của $C$ với cùng eigenvalue $\lambda$.

### 2.5 Eigenfaces (Phép phân rã trị riêng)

Phân rã trị riêng của $L$:

$$L = V \Lambda V^T, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N$$

Khôi phục eigenvectors của $C$ (chuẩn hóa về unit norm):

$$\mathbf{u}_i = \frac{\Phi^T \mathbf{v}_i}{\|\Phi^T \mathbf{v}_i\|}, \quad i = 1, 2, \ldots, k$$

Ma trận Eigenfaces:

$$U_k = [\mathbf{u}_1 | \mathbf{u}_2 | \cdots | \mathbf{u}_k] \in \mathbb{R}^{p \times k}$$

Các cột của $U_k$ tạo thành hệ **trực chuẩn** (orthonormal): $U_k^T U_k = I_k$.

### 2.6 Phép chiếu vuông góc (trọng tâm)

Chiếu ảnh mới $\mathbf{y}$ xuống không gian con $\text{span}(\mathbf{u}_1, \ldots, \mathbf{u}_k)$:

$$\hat{\mathbf{y}} = U_k^T(\mathbf{y} - \bar{\mathbf{x}}) \in \mathbb{R}^k$$

**Ý nghĩa hình học:** $\hat{\mathbf{y}}$ là tọa độ của $(\mathbf{y} - \bar{\mathbf{x}})$ trong hệ cơ sở Eigenface. Phép chiếu là **vuông góc** vì $U_k^T U_k = I$.

### 2.7 Nhận dạng (1-Nearest Neighbor)

$$\text{person}^* = \arg\min_i \|\hat{\mathbf{y}} - \hat{\mathbf{x}}_i\|_2$$

### 2.8 Tái tạo ảnh

$$\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

Sai số tái tạo là phần của $(\mathbf{x} - \bar{\mathbf{x}})$ **vuông góc** với $\text{span}(U_k)$.

### 2.9 Chỉ số chất lượng

$$\text{MSE} = \frac{1}{p}\sum_{i=1}^{p}(\hat{x}_i - x_i)^2, \qquad \text{PSNR} = 10\log_{10}\!\left(\frac{255^2}{\text{MSE}}\right) \text{ [dB]}$$

---

## 3. Cấu trúc dự án

```
_DSTT/
│
├── main_projection.py          # Entry point: chạy toàn bộ pipeline (Bước 1–4, 6)
├── manual_example.py           # Bước 5: ví dụ tính tay trên dữ liệu 4 ảnh 3×3
├── extended_applications.py    # Bước 6: tái tạo ảnh và làm mờ ảnh
├── requirements.txt            # Thư viện cần cài
│
├── src/
│   ├── __init__.py
│   ├── dataloader.py           # Bước 1: tải dataset, vector hóa ảnh, chia train/test
│   ├── recognizer.py           # Bước 2: class OrthogonalFaceRecognizer (core algorithm)
│   ├── baseline.py             # Bước 3: class PixelKNNRecognizer (baseline so sánh)
│   ├── evaluator.py            # Bước 3: accuracy, so sánh k, báo cáo kết quả
│   └── visualizer.py           # Bước 4: 5 loại biểu đồ cho báo cáo và slide
│
├── data/
│   └── orl_faces/              # Dataset ORL (tải tự động khi chạy lần đầu)
│       ├── s1/  (1.pgm … 10.pgm)
│       ├── s2/
│       └── … s40/
│
├── outputs/                    # Toàn bộ biểu đồ được lưu ở đây
│
├── AI_IMPLEMENTATION_GUIDE.md  # Hướng dẫn hiện thực chi tiết
└── REQUIREMENT.md              # Đề tài bài tập lớn
```

---

## 4. Cài đặt

### Yêu cầu hệ thống

- Python ≥ 3.8
- Kết nối internet (lần đầu chạy để tải dataset ~1.7 MB)

### Cài thư viện

```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:

```
numpy
matplotlib
Pillow
opencv-python
```

> **Lưu ý:** Dự án **không dùng** `scikit-learn` hay bất kỳ thư viện ML nào cho phần thuật toán cốt lõi.

---

## 5. Hướng dẫn chạy

### Chạy toàn bộ pipeline (khuyến nghị)

```bash
python main_projection.py
```

Pipeline thực hiện theo thứ tự:

```
[Bước 1] Tải ORL Faces → vector hóa → chia train/test
[Bước 3a] Thử k = 5, 10, 20, 30, 50, 75, 100, 150 → chọn k tối ưu
[Bước 2] Huấn luyện OrthogonalFaceRecognizer với k tối ưu
[Bước 3b] So sánh Eigenfaces vs Baseline Pixel-KNN → in báo cáo
[Bước 4] Sinh 5 biểu đồ → outputs/
[Bước 6] Tái tạo ảnh + làm mờ ảnh → thêm 3 biểu đồ → outputs/
```

### Chạy ví dụ tính tay (Bước 5)

```bash
python manual_example.py
```

In từng bước tính tay trên 4 ảnh 3×3 pixel, xuất `outputs/manual_example.png`.

### Chạy riêng từng module

```bash
python src/dataloader.py           # kiểm tra đọc dataset
python src/recognizer.py           # huấn luyện + đánh giá nhanh
python src/evaluator.py            # so sánh đầy đủ
python src/visualizer.py           # sinh toàn bộ biểu đồ
python extended_applications.py    # tái tạo ảnh và làm mờ ảnh
```

---

## 6. Mô tả chi tiết từng module

### `src/dataloader.py` — Bước 1

| Hàm | Mô tả |
|---|---|
| `download_orl_dataset(save_dir)` | Tải và giải nén ORL Faces. Thử URL chính, nếu lỗi thử URL dự phòng, nếu vẫn lỗi in hướng dẫn tải thủ công |
| `load_image_as_vector(path)` | Mở ảnh → grayscale → resize (92×112) → flatten thành vector `(10304,)` |
| `load_orl_dataset(data_dir, ...)` | Đọc toàn bộ 400 ảnh, chia train/test (mặc định: 2 ảnh cuối/người → test) |
| `print_dataset_info(...)` | In thống kê dataset: shape, số người, giá trị pixel |

**Quy tắc chia train/test:** 8 ảnh đầu/người → train (320 ảnh), 2 ảnh cuối/người → test (80 ảnh).

---

### `src/recognizer.py` — Bước 2

Class `OrthogonalFaceRecognizer` — thuật toán cốt lõi, toàn bộ từ numpy.

| Phương thức | Công thức | Ghi chú |
|---|---|---|
| `fit(X_train, y_train)` | 7 bước pipeline | Lưu mean, eigenfaces, train projections |
| `project(X)` | $\hat{\mathbf{y}} = U_k^T(\mathbf{x} - \bar{\mathbf{x}})$ | **Phép chiếu vuông góc cốt lõi** |
| `reconstruct(X, n_components)` | $\hat{\mathbf{x}} = U_k U_k^T(\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$ | Dùng cho tái tạo và làm mờ ảnh |
| `predict(X_test)` | 1-NN trên không gian eigenface | Khoảng cách Euclidean |
| `explained_variance_ratio()` | $\lambda_i / \sum\lambda$ | Tỉ lệ phương sai giải thích |
| `n_components_for_variance(target)` | — | Tìm k tối thiểu đạt ngưỡng (vd: 95%) |
| `print_summary()` | — | In tóm tắt mô hình |

**Lưu ý kỹ thuật:** Dùng `numpy.linalg.eigh` (thay vì `eig`) vì ma trận hiệp phương sai là thực đối xứng — ổn định số học hơn và đảm bảo trị riêng là số thực.

---

### `src/baseline.py` — Bước 3 (phần 1)

Class `PixelKNNRecognizer` — phương pháp so sánh (không dùng phép chiếu).

| Phương thức | Mô tả |
|---|---|
| `fit(X_train, y_train)` | Lưu tập train (KNN không có bước học thực sự) |
| `predict(X_test)` | k-NN trong không gian pixel $\mathbb{R}^p$ — tính khoảng cách Euclidean trực tiếp |

**Điểm khác biệt so với Eigenfaces:**

| | Eigenfaces | Pixel-KNN |
|---|---|---|
| Không gian so sánh | $\mathbb{R}^k$ ($k \ll p$) | $\mathbb{R}^p$ ($p = 10304$) |
| Bước feature learning | Có (eigendecomposition) | Không |
| Tốc độ predict | Nhanh | Chậm |
| Độ chính xác | Cao hơn | Thấp hơn (nhạy cảm với ánh sáng, nhiễu) |

---

### `src/evaluator.py` — Bước 3 (phần 2)

| Hàm | Mô tả |
|---|---|
| `accuracy(y_true, y_pred)` | Tỉ lệ dự đoán đúng tổng thể |
| `per_class_accuracy(y_true, y_pred)` | Accuracy theo từng person ID |
| `compare_k_values(X_train, y_train, X_test, y_test, k_values)` | Huấn luyện Eigenfaces với nhiều giá trị k, in bảng accuracy + thời gian |
| `run_full_comparison(X_train, y_train, X_test, y_test, best_k)` | Chạy cả hai phương pháp, in báo cáo so sánh đầy đủ |

---

### `src/visualizer.py` — Bước 4

| Hàm | File output | Nội dung |
|---|---|---|
| `plot_mean_face(recognizer)` | `mean_face.png` | Khuôn mặt trung bình $\bar{\mathbf{x}}$ |
| `plot_eigenfaces(recognizer, n_faces)` | `eigenfaces.png` | Lưới top-20 eigenfaces, mỗi ảnh ghi % variance |
| `plot_recognition_examples(...)` | `recognition.png` | 1 ví dụ nhận dạng đúng (viền xanh) + 1 sai (viền đỏ) |
| `plot_accuracy_vs_k(k_results, baseline_acc)` | `accuracy_vs_k.png` | Đường cong accuracy theo k + đường baseline |
| `plot_explained_variance(recognizer)` | `variance_ratio.png` | Cột variance từng eigenface + đường tích lũy + ngưỡng 95% |
| `plot_all(...)` | Tất cả 5 file | Gọi tất cả hàm trên cùng lúc |

---

### `manual_example.py` — Bước 5

Minh họa từng bước thuật toán trên **dữ liệu đồ chơi nhỏ** (4 ảnh 3×3 pixel, 2 người):

| Bước | Nội dung in ra |
|---|---|
| 1 | Từng vector ảnh $\mathbf{x}_1, \ldots, \mathbf{x}_4$ và $\mathbf{x}_\text{test}$ |
| 2 | Tính từng phần tử của $\bar{\mathbf{x}}$ (hiện phép cộng và chia) |
| 3 | Từng $\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}$ và ma trận $\Phi$ đầy đủ |
| 4 | Từng $L[i,j] = \Phi_i \cdot \Phi_j / N$ và giải thích thủ thuật |
| 5 | Eigenvalues gốc (tăng dần) → sau sắp xếp giảm dần + % variance |
| 6 | $\Phi^T \mathbf{v}_i$ chưa chuẩn hóa → norm → $\mathbf{u}_i$ + kiểm tra $U^T U = I$ |
| 7 | Từng $\hat{\mathbf{y}}_i = U^T(\mathbf{x}_i - \bar{\mathbf{x}})$ cho tập train |
| 8 | Chiếu $\mathbf{x}_\text{test}$ → khoảng cách từng bước → nhãn dự đoán |
| 9 | Biểu đồ: ảnh train, eigenfaces, scatter 2D trong không gian eigenface |

**Mục đích:** Cung cấp ví dụ cụ thể có thể đối chiếu thủ công cho phần Thực hành của báo cáo.

---

### `extended_applications.py` — Bước 6

#### Ứng dụng 1: Tái tạo ảnh

Nén ảnh bằng cách chỉ lưu $k$ tọa độ eigenface thay vì $p = 10304$ pixel:

$$\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

| Hàm | File output | Nội dung |
|---|---|---|
| `plot_reconstruction_comparison(...)` | `app1_reconstruction.png` | Lưới: gốc \| k=1 \| k=5 \| k=20 \| k=50 \| k=100 \| k=150 — kèm MSE, PSNR |
| `plot_reconstruction_quality(...)` | `app1_quality_curve.png` | Đường cong MSE ↓ và PSNR ↑ theo k |

#### Ứng dụng 2: Làm mờ ảnh

Các eigenface đầu tiên (eigenvalue lớn) nắm bắt **tần số thấp** (cấu trúc tổng thể), các eigenface sau nắm bắt **tần số cao** (chi tiết nhỏ, cạnh). Tái tạo ảnh chỉ với $k$ eigenface đầu tiên hoạt động như một **bộ lọc thông thấp** (low-pass filter): loại bỏ chi tiết tần số cao ⇒ ảnh bị làm mờ. Mức độ mờ tăng khi $k$ giảm.

| Hàm | File output | Nội dung |
|---|---|---|
| `plot_blurring_effect(...)` | `app2_blurring.png` | Lưới: gốc \| k=1 \| k=3 \| k=7 \| k=15 \| k=30 — kèm MSE, PSNR. k nhỏ ⇒ ảnh mờ hơn |

---

## 7. Kết quả kỳ vọng

### Accuracy nhận dạng

| Phương pháp | Accuracy (kỳ vọng) |
|---|---|
| Eigenfaces (k tối ưu ~50) | 92–97% |
| Baseline Pixel-KNN (1-NN) | 80–88% |
| Eigenfaces cải thiện | +5–15% |

### Chất lượng tái tạo (PSNR)

| k eigenfaces | PSNR (dB) | Chất lượng |
|---|---|---|
| 1 | ~3 | Kém (chỉ thấy hình bóng) |
| 5 | ~10 | Nhận ra khuôn mặt mờ |
| 20 | ~25 | Chấp nhận được |
| 50 | ~35 | Tốt |
| 100 | ~42 | Rất tốt |
| 150 | ~48 | Gần như hoàn hảo |

---

## 8. Biểu đồ đầu ra

Sau khi chạy `python main_projection.py`, thư mục `outputs/` chứa:

```
outputs/
├── mean_face.png           # Khuôn mặt trung bình x̄
├── eigenfaces.png          # Lưới top-20 eigenfaces (% variance mỗi ảnh)
├── recognition.png         # Ví dụ nhận dạng đúng (xanh) và sai (đỏ)
├── accuracy_vs_k.png       # Accuracy vs k + đường baseline
├── variance_ratio.png      # Phương sai giải thích từng eigenface + tích lũy
│
├── app1_reconstruction.png # So sánh tái tạo: gốc | k=1 | k=5 | … | k=150
├── app1_quality_curve.png  # Đường cong MSE và PSNR theo k
│
└── app2_blurring.png       # Hiệu ứng làm mờ với k nhỏ dần (k=1, 3, 7, 15, 30)
```

Sau khi chạy `python manual_example.py`:

```
outputs/
└── manual_example.png      # Ảnh train + eigenfaces + scatter 2D eigenface space
```

---

## 9. Tài liệu tham khảo

```
[1] Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition."
    Journal of Cognitive Neuroscience, 3(1), 71–86.

[2] Jolliffe, I. T. (2002). "Principal Component Analysis." 2nd ed. Springer.

[3] Gonzalez, R. C. & Woods, R. E. (2018). "Digital Image Processing."
    4th ed. Pearson.  (định nghĩa MSE, PSNR trong xử lý ảnh số)

[4] Strang, G. (2016). "Introduction to Linear Algebra." 5th ed. Wellesley-Cambridge Press.
    (cơ sở lý thuyết về phép chiếu vuông góc, eigendecomposition)

[5] Cover, T. & Hart, P. (1967). "Nearest Neighbor Pattern Classification."
    IEEE Transactions on Information Theory, 13(1), 21–27.

[6] AT&T Laboratories Cambridge. "The ORL Database of Faces."
    https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

[7] NumPy documentation — numpy.linalg.eigh.
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html

[8] Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment."
    Computing in Science & Engineering, 9(3), 90–95.
```
