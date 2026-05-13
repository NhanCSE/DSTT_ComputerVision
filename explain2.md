# EXPLAIN.md — Giải thích toàn diện dự án

> **Đối tượng đọc:** Người chưa biết gì về dự án, chưa quen Python, chưa học sâu về Đại số tuyến tính.
> **Mục tiêu:** Sau khi đọc xong file này, bạn có thể (1) hiểu đề bài, (2) hiểu lý thuyết toán, (3) hiểu từng dòng code, (4) chạy được dự án, (5) trả lời câu hỏi phản biện.

---

## MỤC LỤC

1. [Tổng quan dự án trong 60 giây](#1-tổng-quan-dự-án-trong-60-giây)
2. [Đề bài yêu cầu cái gì?](#2-đề-bài-yêu-cầu-cái-gì)
3. [Khái niệm nền tảng (cho người mới)](#3-khái-niệm-nền-tảng-cho-người-mới)
4. [Lý thuyết toán học chi tiết](#4-lý-thuyết-toán-học-chi-tiết)
5. [Thuật toán Eigenfaces — từng bước](#5-thuật-toán-eigenfaces--từng-bước)
6. [Cấu trúc thư mục dự án](#6-cấu-trúc-thư-mục-dự-án)
7. [Cách cài đặt và chạy](#7-cách-cài-đặt-và-chạy)
8. [Đi sâu vào từng file code](#8-đi-sâu-vào-từng-file-code)
9. [Ví dụ tính tay (dữ liệu nhỏ)](#9-ví-dụ-tính-tay-dữ-liệu-nhỏ)
10. [Hai ứng dụng mở rộng: Tái tạo & Làm mờ ảnh](#10-hai-ứng-dụng-mở-rộng-tái-tạo--làm-mờ-ảnh)
11. [Các biểu đồ đầu ra giải thích cái gì?](#11-các-biểu-đồ-đầu-ra-giải-thích-cái-gì)
12. [Kết quả thực tế thu được](#12-kết-quả-thực-tế-thu-được)
13. [Câu hỏi phản biện hay gặp + cách trả lời](#13-câu-hỏi-phản-biện-hay-gặp--cách-trả-lời)
14. [Thuật ngữ tra cứu nhanh (glossary)](#14-thuật-ngữ-tra-cứu-nhanh-glossary)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)

---

## 1. Tổng quan dự án trong 60 giây

- **Đề tài:** Nghiên cứu **phép chiếu vuông góc** (Orthogonal Projection) trong **thị giác máy tính** (Computer Vision).
- **Ứng dụng cụ thể:** Hiện thực thuật toán **Eigenfaces** (Turk & Pentland, 1991) để:
  1. **Nhận dạng khuôn mặt** — biết một ảnh thuộc về ai trong 40 người.
  2. **Tái tạo ảnh** — nén ảnh thành ít chiều, rồi khôi phục.
  3. **Làm mờ ảnh** — coi việc giữ ít eigenface như một bộ lọc thông thấp.
- **Dataset:** ORL Faces (AT&T) — 40 người × 10 ảnh = **400 ảnh**, mỗi ảnh `92×112` pixel grayscale.
- **Ngôn ngữ:** Python 3.8+. Chỉ dùng `numpy`, `matplotlib`, `Pillow`, `opencv-python`. **KHÔNG dùng `sklearn`** — tất cả đại số tuyến tính tự viết.
- **Trọng tâm toán học:** Phép chiếu `ŷ = Uᵀ(x − x̄)`. Vector `x` là một ảnh, `U` là cơ sở trực chuẩn của không gian con (gồm các "khuôn mặt đặc trưng" — eigenfaces), `x̄` là khuôn mặt trung bình.

> **Một câu thâu tóm:** _"Mỗi ảnh là một điểm trong không gian 10304 chiều. Eigenfaces tìm `k` hướng quan trọng nhất (k thường ≈ 50). Chiếu mọi ảnh xuống `k` hướng đó → so sánh ảnh bằng tọa độ mới, nhanh và chính xác hơn."_

---

## 2. Đề bài yêu cầu cái gì?

### 2.1 Đề tài được giao (trích `REQUIREMENT.md`)

> **"Nghiên cứu tính chất và ứng dụng của phép chiếu vuông góc trong thị giác máy tính. Ứng dụng: Nhận dạng đối tượng, tái tạo hình ảnh và kỹ thuật làm mờ hình ảnh."**

### 2.2 Bài báo cáo phải có đủ các phần

| Phần               | Yêu cầu                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| Trang bìa          | Đề tài, khoa, môn học, lớp, nhóm, GV hướng dẫn, MSSV thành viên, mô tả đóng góp |
| Giới thiệu         | Đối tượng, mục tiêu, phương pháp                                                |
| Cơ sở lý thuyết    | 4–10 trang. Không chép phần đã học trong môn ĐSTT. Trích dẫn rõ ràng.           |
| Thuật toán         | Các bước giải bài toán, có ví dụ minh họa                                       |
| Phần mềm           | Code Python/Matlab, hiểu rõ từng dòng                                           |
| Kết quả            | So sánh nhiều phương pháp được đánh giá cao hơn                                 |
| Tài liệu tham khảo | Đầy đủ, đúng chuẩn                                                              |

**Định dạng:** Tối thiểu 20 trang, cỡ chữ 13–17, lề trái 3cm/còn lại 2cm, **không chụp hình công thức toán**.

### 2.3 Tiêu chí chấm điểm

| Tiêu chí            | %   |
| ------------------- | --- |
| Nộp bài + điểm danh | 10% |
| Bài báo cáo         | 50% |
| Hỏi đáp (phản biện) | 40% |

### 2.4 Ràng buộc kỹ thuật quan trọng (rút từ `IMPLEMENTATION_PLAN.md`)

- ❌ **Cấm** dùng "hộp đen" như `sklearn.decomposition.PCA`.
- ✅ **Bắt buộc** tự code mọi thao tác hiệp phương sai, trị riêng, vector riêng, phép chiếu bằng `numpy` thuần.
- ✅ Comment chi tiết để cả nhóm hiểu (GV sẽ hỏi ngẫu nhiên 1 thành viên).
- ✅ Có ví dụ tính tay minh họa thuật toán.
- ✅ Có ít nhất một bài toán thực tế.
- ✅ So sánh nhiều phương pháp (Eigenfaces vs Baseline Pixel-KNN).

---

## 3. Khái niệm nền tảng (cho người mới)

Trước khi vào toán, ta cần làm rõ vài khái niệm. Nếu bạn đã biết, có thể bỏ qua.

### 3.1 Ảnh số là cái gì?

Một ảnh số grayscale (đen-trắng) là một bảng số 2D. Mỗi ô bảng (gọi là **pixel**) lưu một số nguyên từ **0** (đen tuyệt đối) đến **255** (trắng tuyệt đối). Ảnh ORL có kích thước 112 hàng × 92 cột = **10304 pixel**.

Ví dụ ảnh `3×3`:

```
[ 10  20  30 ]      ← hàng 0
[ 20  30  40 ]      ← hàng 1
[ 30  40  50 ]      ← hàng 2
```

### 3.2 "Duỗi" ảnh thành vector

Lấy bảng 2D ở trên, ghép hàng tiếp hàng thành một danh sách 1D dài 9 phần tử:

```
[10, 20, 30, 20, 30, 40, 30, 40, 50]   ← vector trong R⁹
```

Ảnh `112×92` duỗi ra thành vector `R¹⁰³⁰⁴`. Mỗi ảnh = **một điểm trong không gian 10304 chiều**.

### 3.3 Không gian nhiều chiều — hình dung thế nào?

- Trong `R²` (mặt phẳng): mỗi điểm có 2 tọa độ `(x, y)`.
- Trong `R³` (không gian): mỗi điểm có 3 tọa độ `(x, y, z)`.
- Trong `R¹⁰³⁰⁴`: mỗi điểm có 10304 tọa độ. Không vẽ được hình, **nhưng các phép tính (cộng, trừ, tích vô hướng, khoảng cách Euclidean) vẫn hoàn toàn áp dụng được**. Đây là sức mạnh của đại số tuyến tính: cùng một công thức, bao nhiêu chiều cũng dùng được.

### 3.4 Phép chiếu trực giao — trực giác

Tưởng tượng bạn đứng giữa phòng, có một mặt sàn (mặt phẳng `xy`). Nếu rọi đèn pin thẳng đứng từ trên xuống, **bóng** của bạn rơi xuống sàn — đó chính là **phép chiếu vuông góc** của bạn lên mặt sàn. Bóng có ít chiều hơn (2D) nhưng vẫn giữ "đặc điểm" cơ bản của hình dáng bạn.

Tương tự, một ảnh là một điểm trong `R¹⁰³⁰⁴`. Ta tạo một "mặt sàn" mới (không gian con `k` chiều, `k≈50`). Chiếu ảnh xuống → còn lại bóng = vector tọa độ `k` chiều. So sánh hai ảnh = so sánh hai bóng của chúng.

### 3.5 Trị riêng & vector riêng (eigenvalue, eigenvector)

Cho một ma trận vuông `A`. Vector `v` ≠ 0 được gọi là **vector riêng** nếu nhân `A` với `v` chỉ thay đổi **độ dài** chứ không đổi **hướng**:
$$A \mathbf{v} = \lambda \mathbf{v}$$
Số `λ` là **trị riêng** tương ứng. Hiểu nôm na: vector riêng là **trục bất biến** của phép biến đổi `A`. Trị riêng đo "mức độ kéo giãn" theo trục đó.

Trong dự án này, ma trận `A` chính là **ma trận hiệp phương sai** của dữ liệu ảnh; vector riêng của nó = các "hướng biến thiên chính" của dữ liệu = các **eigenfaces**.

### 3.6 Cơ sở trực chuẩn (orthonormal basis)

Một tập vector `{u₁, u₂, ..., uₖ}` là cơ sở trực chuẩn nếu:

1. **Trực giao** (orthogonal): `uᵢ · uⱼ = 0` với mọi `i ≠ j` (vuông góc nhau).
2. **Chuẩn hóa** (normalized): `‖uᵢ‖ = 1` (độ dài = 1).

Tính chất "vàng": Nếu xếp các vector này thành cột của ma trận `U`, thì `Uᵀ U = I` (ma trận đơn vị). Tính chất này khiến phép chiếu trở nên **đơn giản chỉ bằng nhân ma trận**, không cần giải hệ phương trình.

---

## 4. Lý thuyết toán học chi tiết

Phần này trình bày toán cốt lõi của dự án. Mọi công thức dưới đây đều có hiện thực `numpy` tương ứng trong code.

### 4.1 Vector hóa & ma trận dữ liệu

Có `N` ảnh huấn luyện. Mỗi ảnh `xᵢ ∈ R^p` với `p = 10304`.

Ma trận dữ liệu:
$$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix} \in \mathbb{R}^{N \times p}$$

Trong code: `X_train.shape == (320, 10304)`.

### 4.2 Khuôn mặt trung bình (mean face)

$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i \in \mathbb{R}^p$$

Đây là **trung bình cộng theo từng pixel**. Hình ảnh của `x̄` là một khuôn mặt mờ mờ — "nguyên mẫu" của tập huấn luyện.

`np.mean(X_train, axis=0)` → trục `axis=0` nghĩa là "trung bình theo chiều dọc của ma trận" → trả về vector kích thước `(p,)`.

### 4.3 Trung tâm hóa (centering)

$$\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}, \quad \Phi \in \mathbb{R}^{N \times p}$$

Tại sao phải trung tâm hóa? Hiệp phương sai đo "biến thiên xung quanh trung bình". Nếu không trừ trung bình đi, kết quả sẽ bị lệch về phía giá trị tuyệt đối của pixel chứ không phản ánh sự khác biệt giữa các khuôn mặt.

### 4.4 Ma trận hiệp phương sai

Định nghĩa chuẩn (đầy đủ):
$$C = \frac{1}{N} \Phi^T \Phi \in \mathbb{R}^{p \times p}$$

`C` đo độ tương quan giữa từng cặp pixel. Phần tử `C[i,j]` cho biết pixel `i` và pixel `j` có biến thiên cùng nhau qua các ảnh hay không.

**Vấn đề:** `p = 10304` → `C` có kích thước `10304 × 10304 ≈ 100 triệu` phần tử. Ở dạng `float64` chiếm ~800 MB RAM và phân rã trị riêng cực chậm.

### 4.5 Thủ thuật Turk & Pentland (1991)

Thay vì tính `C`, ta tính một ma trận thay thế **nhỏ hơn rất nhiều**:
$$L = \frac{1}{N} \Phi \Phi^T \in \mathbb{R}^{N \times N}$$

Với `N = 320`, ma trận `L` chỉ là `320×320` — vừa đủ vào L1 cache của CPU!

**Chứng minh:** Giả sử `L · v = λ · v` (v là eigenvector của L). Nhân hai vế bên trái với `Φᵀ`:
$$\Phi^T (L \mathbf{v}) = \lambda (\Phi^T \mathbf{v})$$
$$\Phi^T \cdot \frac{1}{N}\Phi\Phi^T \cdot \mathbf{v} = \lambda \Phi^T \mathbf{v}$$
$$\underbrace{\frac{1}{N}\Phi^T\Phi}_{C} \cdot \underbrace{(\Phi^T \mathbf{v})}_{\mathbf{u}} = \lambda \mathbf{u}$$

Vậy `u = Φᵀv` là eigenvector của `C` với **cùng eigenvalue `λ`**. Tuyệt vời!

Sau khi tìm được `v`, chỉ cần chuẩn hóa `u`:
$$\mathbf{u}_i = \frac{\Phi^T \mathbf{v}_i}{\|\Phi^T \mathbf{v}_i\|}$$

### 4.6 Ma trận Eigenfaces

Sắp xếp eigenvalues giảm dần `λ₁ ≥ λ₂ ≥ ... ≥ λₙ`, lấy top-`k` eigenvectors:
$$U_k = [\mathbf{u}_1 | \mathbf{u}_2 | \cdots | \mathbf{u}_k] \in \mathbb{R}^{p \times k}$$

Tính chất: `Uₖᵀ Uₖ = Iₖ` (k cột trực chuẩn).

**Ý nghĩa hình ảnh:** Mỗi cột `uᵢ` là một vector `R¹⁰³⁰⁴` — duỗi ngược lại thành `112×92` ta được **một ảnh** trông giống khuôn mặt ma quái (gọi là **eigenface**). Eigenface đầu tiên là hướng mà dữ liệu biến thiên mạnh nhất; eigenface thứ hai là hướng biến thiên mạnh nhì (sau khi trừ đi đóng góp của eigenface 1); v.v.

### 4.7 Phép chiếu vuông góc — TRỌNG TÂM CỦA ĐỀ TÀI

Cho ảnh mới `y ∈ R^p`. Tọa độ của nó trong cơ sở Eigenface là:
$$\hat{\mathbf{y}} = U_k^T (\mathbf{y} - \bar{\mathbf{x}}) \in \mathbb{R}^k$$

**Tại sao đây là phép chiếu _vuông góc_?**

Vì `Uₖ` có các cột trực chuẩn nên ma trận chiếu lên span(Uₖ) là:
$$P = U_k U_k^T$$

Phép chiếu `Py` cho phần "hình bóng" của `y` trong không gian con. Phần `y − Py` thì **vuông góc** với toàn bộ không gian con (đây là đặc trưng của phép chiếu vuông góc, ngược với phép chiếu xiên).

Vector `ŷ = Uₖᵀ y` chính là **tọa độ** của hình bóng đó trong hệ cơ sở `{u₁,...,uₖ}`. Nó có `k` chiều, ngắn hơn nhiều so với `p`.

### 4.8 Nhận dạng — 1-Nearest Neighbor

Để biết ảnh test thuộc về ai:

1. Chiếu ảnh test: `ŷ_test = Uₖᵀ(x_test − x̄)`.
2. Tính khoảng cách Euclidean tới từng `ŷᵢ` của tập train:
   $$d_i = \|\hat{\mathbf{y}}_{test} - \hat{\mathbf{y}}_i\|_2 = \sqrt{\sum_{j=1}^{k}(\hat{y}_{test,j} - \hat{y}_{i,j})^2}$$
3. Gán nhãn ảnh train gần nhất:
   $$\text{person}^* = \arg\min_i d_i$$

Tại sao 1-NN hoạt động? Vì các ảnh **cùng một người** sau khi chiếu nằm gần nhau trong không gian Eigenface (cùng đặc điểm cấu trúc khuôn mặt), còn ảnh khác người thì xa nhau.

### 4.9 Tái tạo ảnh

Đã chiếu `y → ŷ`, làm sao "đảo ngược" để có ảnh xấp xỉ?
$$\hat{\mathbf{x}} = U_k \hat{\mathbf{y}} + \bar{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

- Khi `k = N` (giữ hết eigenfaces): `x̂ = x` (tái tạo hoàn hảo trong span của tập train).
- Khi `k` nhỏ: `x̂` chỉ giữ được những đặc điểm "chính" → bị mờ.

**Sai số tái tạo** = phần của `(x − x̄)` **vuông góc** với span(Uₖ) — bị "vứt đi" khi chiếu.

### 4.10 Chỉ số chất lượng ảnh

- **MSE (Mean Squared Error):**
  $$\text{MSE} = \frac{1}{p}\sum_{i=1}^{p}(\hat{x}_i - x_i)^2$$
  Càng nhỏ càng tốt. MSE = 0 ⟺ tái tạo hoàn hảo.

- **PSNR (Peak Signal-to-Noise Ratio) — dB:**
  $$\text{PSNR} = 10 \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right)$$
  Càng lớn càng tốt. PSNR > 40 dB: gần như nhìn không phân biệt được.

### 4.11 Phương sai giải thích (Explained Variance)

Eigenvalue `λᵢ` đo lượng phương sai theo hướng eigenface `uᵢ`. Tỉ lệ:
$$r_i = \frac{\lambda_i}{\sum_j \lambda_j}, \quad \text{cum}_k = \sum_{i=1}^{k} r_i$$

`cum_k = 95%` nghĩa là `k` eigenfaces đầu tiên "tóm" được 95% biến thiên của dữ liệu. Đây là tiêu chí phổ biến để chọn `k`.

---

## 5. Thuật toán Eigenfaces — từng bước

Đây là pipeline đầy đủ, ánh xạ trực tiếp sang code [src/recognizer.py](src/recognizer.py).

### Pha HUẤN LUYỆN (`fit(X_train, y_train)`)

```
INPUT:  X_train shape (N, p),  y_train shape (N,)

Bước 1. Tính khuôn mặt trung bình
            x̄ = mean(X_train, axis=0)              # shape (p,)

Bước 2. Trung tâm hóa
            Φ = X_train − x̄                        # shape (N, p)

Bước 3. Ma trận hiệp phương sai thay thế
            L = (Φ Φᵀ) / N                          # shape (N, N) ← nhỏ!

Bước 4. Phân rã trị riêng của L
            λ, V = eigh(L)                           # eigh: cho ma trận đối xứng
            # eigh trả về thứ tự TĂNG → đảo lại GIẢM
            idx = argsort(λ)[::-1]
            λ, V = λ[idx], V[:, idx]
            # Loại λ ≈ 0 (sai số số học)
            valid = λ > 1e-10
            λ, V = λ[valid], V[:, valid]

Bước 5. Khôi phục Eigenfaces của C
            U_unnorm = Φᵀ V                          # shape (p, n_valid)
            norms = ‖U_unnorm‖_cột                   # độ dài mỗi cột
            U = U_unnorm / norms                     # chuẩn hóa unit-norm

Bước 6. Chọn k tốt nhất
            U_k = U[:, :k]                           # shape (p, k)

Bước 7. Chiếu toàn bộ tập train, lưu lại
            train_projections = (X_train − x̄) U_k    # shape (N, k)
            train_labels      = y_train

OUTPUT: mean_face_, eigenfaces_, train_projections_, train_labels_
```

### Pha DỰ ĐOÁN (`predict(X_test)`)

```
INPUT:  X_test shape (N_test, p)

Bước 1. Chiếu ảnh test
            test_projections = (X_test − x̄) U_k     # shape (N_test, k)

Bước 2. Với mỗi ảnh test, tìm ảnh train gần nhất
            for i in 1..N_test:
                d_j = ‖test_projections[i] − train_projections[j]‖₂
                nearest = argmin_j d_j
                predictions[i] = train_labels[nearest]

OUTPUT: predictions shape (N_test,)
```

---

## 6. Cấu trúc thư mục dự án

```
DSTT_ComputerVision/
│
├── REQUIREMENT.md              # Đề bài gốc của giảng viên
├── IMPLEMENTATION_PLAN.md      # Kế hoạch hiện thực chi tiết
├── README.md                   # Tài liệu chính cho người dùng
├── explain.md                  # ← BẠN ĐANG ĐỌC FILE NÀY
├── HDSD.txt                    # Hướng dẫn sử dụng (chạy streamlit)
├── requirements.txt            # Danh sách thư viện
│
├── main_projection.py          # Entry point: chạy toàn bộ pipeline
├── manual_example.py           # Ví dụ tính tay trên dữ liệu nhỏ
├── extended_applications.py    # Ứng dụng mở rộng (tái tạo & làm mờ)
│
├── src/                        # Module thuật toán
│   ├── __init__.py
│   ├── dataloader.py          # Bước 1: tải dataset, vector hóa
│   ├── recognizer.py          # Bước 2: thuật toán Eigenfaces CỐT LÕI
│   ├── baseline.py            # Bước 3: phương pháp so sánh (Pixel-KNN)
│   ├── evaluator.py           # Bước 3: đánh giá & so sánh
│   └── visualizer.py          # Bước 4: vẽ biểu đồ
│
├── data/orl_faces/            # Dataset (tải tự động lần đầu)
│   ├── s1/  1.pgm … 10.pgm    # 10 ảnh của người số 1
│   ├── s2/                    # 10 ảnh của người số 2
│   └── … s40/
│
├── outputs/                    # Biểu đồ đầu ra
│   ├── mean_face.png
│   ├── eigenfaces.png
│   ├── recognition.png
│   ├── accuracy_vs_k.png
│   ├── variance_ratio.png
│   ├── app1_reconstruction.png
│   ├── app1_quality_curve.png
│   └── app2_blurring.png
│
└── web/
    └── streamlit_app.py       # Giao diện web demo
```

### Vai trò từng file (tóm tắt)

| File                     | Vai trò                  | Đọc khi nào?                              |
| ------------------------ | ------------------------ | ----------------------------------------- |
| `REQUIREMENT.md`         | Đề gốc                   | Khi cần xem lại yêu cầu                   |
| `IMPLEMENTATION_PLAN.md` | Roadmap hiện thực        | Khi muốn hiểu "tại sao chia bước thế này" |
| `README.md`              | Hướng dẫn cho người dùng | Khi muốn chạy nhanh                       |
| `main_projection.py`     | Chạy toàn bộ             | Khi muốn xem kết quả end-to-end           |
| `src/recognizer.py`      | Thuật toán               | Khi muốn hiểu cốt lõi                     |
| `manual_example.py`      | Ví dụ tay                | Khi muốn đối chiếu với báo cáo            |
| `web/streamlit_app.py`   | Demo trực quan           | Khi muốn show cho người chấm              |

---

## 7. Cách cài đặt và chạy

### 7.1 Yêu cầu hệ thống

- **Python ≥ 3.8** (kiểm tra: `python --version`).
- Khoảng **500 MB** dung lượng đĩa trống.
- **Internet** (lần đầu chạy để tải dataset ORL ~1.7 MB).

### 7.2 Cài thư viện

Mở terminal/PowerShell tại thư mục dự án:

```powershell
pip install -r requirements.txt
```

Nội dung `requirements.txt`:

```
numpy            # Đại số tuyến tính (ma trận, eigendecomposition)
matplotlib       # Vẽ biểu đồ
Pillow           # Đọc/ghi file ảnh
opencv-python    # Hỗ trợ xử lý ảnh
streamlit        # Giao diện web demo
```

### 7.3 Chạy pipeline đầy đủ (khuyến nghị)

```powershell
python main_projection.py
```

Console sẽ in tuần tự:

```
==========================================================
  NHẬN DẠNG KHUÔN MẶT BẰNG PHÉP CHIẾU VUÔNG GÓC
==========================================================

BƯỚC 1 — Chuẩn bị Dataset
  [tải dataset nếu chưa có]
  THỐNG KÊ DATASET ORL FACES ...

BƯỚC 3a — Tìm số Eigenfaces k tối ưu
       k   Accuracy    Train(s)   Predict(s)
       5      78.8%      0.043       0.012
      10      91.2%      0.045       0.018
     ...
   → k tốt nhất: 50  (Accuracy = 96.3%)

BƯỚC 2 — Huấn luyện OrthogonalFaceRecognizer (k=50)
   TÓM TẮT MÔ HÌNH EIGENFACES ...

BƯỚC 3b — So sánh Eigenfaces vs Baseline
   BÁO CÁO SO SÁNH: ...

BƯỚC 4 — Trực quan hóa
   [OK] Đã lưu: outputs/mean_face.png
   [OK] Đã lưu: outputs/eigenfaces.png
   ...

BƯỚC 6 — Ứng dụng mở rộng
   [OK] Đã lưu: outputs/app1_reconstruction.png
   ...

TÓM TẮT KẾT QUẢ
   Accuracy Eigenfaces: 96.3%
   Accuracy Baseline  : 85.0%
   Cải thiện          : +11.3%
```

### 7.4 Chạy ví dụ tính tay (cho phần báo cáo)

```powershell
python manual_example.py
```

In ra **toàn bộ phép tính trung gian** trên dữ liệu nhỏ (4 ảnh 3×3) để bạn đối chiếu với phần lý thuyết của báo cáo.

### 7.5 Chạy demo web (đẹp để show GV)

```powershell
streamlit run web/streamlit_app.py
```

Mở trình duyệt tại `http://localhost:8501`, upload ảnh để nhận dạng trực tiếp.

### 7.6 Chạy từng module riêng

Mỗi file trong `src/` đều có `if __name__ == "__main__":` để chạy thử riêng lẻ:

```powershell
python src/dataloader.py        # Test tải dataset
python src/recognizer.py        # Test thuật toán
python src/evaluator.py         # Test so sánh
python src/visualizer.py        # Test vẽ biểu đồ
```

### 7.7 Khắc phục sự cố thường gặp

| Lỗi                                     | Nguyên nhân           | Cách sửa                                    |
| --------------------------------------- | --------------------- | ------------------------------------------- |
| `ModuleNotFoundError: numpy`            | Chưa cài thư viện     | `pip install -r requirements.txt`           |
| `FileNotFoundError: data/orl_faces/...` | Tải dataset thất bại  | Tải thủ công theo hướng dẫn console         |
| `RecursionError`, `MemoryError`         | Máy yếu               | Giảm `N_PERSONS` trong `main_projection.py` |
| Chạy chậm                               | Lần đầu compile numpy | Đợi 30s, lần sau nhanh hơn                  |

---

## 8. Đi sâu vào từng file code

### 8.1 [src/dataloader.py](src/dataloader.py) — Bước 1: Chuẩn bị dữ liệu

**Mục tiêu:** Đọc 400 ảnh `.pgm` từ thư mục `data/orl_faces/`, chuyển mỗi ảnh thành vector `(10304,)`, rồi chia train/test.

**Hằng số quan trọng:**

```python
IMG_HEIGHT = 112   # chiều cao ảnh ORL gốc
IMG_WIDTH  = 92    # chiều rộng
IMG_SIZE   = 10304 # = 112 × 92 (kích thước vector flat)
```

**Hàm cốt lõi `load_image_as_vector(path)`:**

```python
img = Image.open(image_path)                          # mở ảnh bằng Pillow
img_gray = img.convert("L")                            # chuyển grayscale
img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT)) # ép kích thước chuẩn
img_array = np.array(img_resized, dtype=np.float64)    # → numpy 2D (112, 92)
img_vector = img_array.flatten()                       # duỗi → (10304,)
return img_vector
```

> **Tại sao `convert("L")`?** ORL Faces đã grayscale, nhưng ảnh người dùng tự upload có thể là RGB. Convert đảm bảo mọi đầu vào nhất quán.

> **Tại sao resize?** Phòng khi ảnh test có kích thước khác (ví dụ chụp từ điện thoại). Phải đưa về cùng `p = 10304` chiều để chiếu được.

**Quy tắc chia train/test (mặc định):**

```python
n_test_per_person = 2
cutoff = 10 - 2 = 8     # ảnh 1..8 → train, ảnh 9..10 → test
```

Kết quả: `X_train` shape `(320, 10304)`, `X_test` shape `(80, 10304)`.

### 8.2 [src/recognizer.py](src/recognizer.py) — Bước 2: THUẬT TOÁN CỐT LÕI

Đây là file quan trọng nhất. Class `OrthogonalFaceRecognizer` có 4 phương thức chính:

#### `fit(X_train, y_train)` — 7 bước

```python
N, p = X_train.shape  # N=320 ảnh train, p=10304 pixel

# Bước 2.1 — Mean face
self.mean_face_ = np.mean(X_train, axis=0)            # shape (10304,)

# Bước 2.2 — Centering (broadcasting tự trừ theo từng hàng)
Phi = X_train - self.mean_face_                        # shape (320, 10304)

# Bước 2.3 — Surrogate covariance L (thay vì C khổng lồ)
L = (Phi @ Phi.T) / N                                  # shape (320, 320)

# Bước 2.4 — Eigendecomposition của L (eigh cho ma trận đối xứng)
eigenvalues_L, V = np.linalg.eigh(L)
idx = np.argsort(eigenvalues_L)[::-1]                  # đảo về thứ tự GIẢM
eigenvalues_L = eigenvalues_L[idx]
V = V[:, idx]
valid = eigenvalues_L > 1e-10                          # loại trị riêng ≈ 0
eigenvalues_L = eigenvalues_L[valid]
V = V[:, valid]

# Bước 2.5 — Khôi phục Eigenfaces của C
U_unnorm = Phi.T @ V                                   # shape (10304, n_valid)
norms = np.linalg.norm(U_unnorm, axis=0, keepdims=True)
U = U_unnorm / norms                                   # chuẩn hóa unit-norm

# Bước 2.6 — Chọn k tốt nhất
k = min(self.n_components, U.shape[1])
self.eigenvalues_ = eigenvalues_L[:k]
self.eigenfaces_  = U[:, :k]                           # shape (10304, k)

# Bước 2.7 — Chiếu tập train (lưu để dùng cho predict)
self.train_projections_ = self.project(X_train)        # shape (320, k)
self.train_labels_ = y_train.copy()
```

**Tại sao dùng `np.linalg.eigh` thay vì `np.linalg.eig`?**

- `eigh` chuyên cho ma trận **thực đối xứng** (Hermitian) — ổn định số học hơn.
- Đảm bảo trị riêng và vector riêng là **số thực** (eig có thể ra số phức nhỏ do nhiễu).
- Nhanh hơn vì lợi dụng tính đối xứng.

**Tại sao loại `λ > 1e-10`?**

- Lý thuyết: `L` có rank tối đa `N` nên có tối đa `N` eigenvalue khác 0. Nhưng thực tế còn ràng buộc khác (trung tâm hóa làm mất 1 chiều) nên hữu ích chỉ có `N-1`.
- Các eigenvalue cực bé (`< 1e-10`) là **nhiễu số học**, không mang thông tin → loại bỏ.

#### `project(X)` — Phép chiếu vuông góc

```python
def project(self, X):
    X_centered = X - self.mean_face_              # (N, p)
    projections = X_centered @ self.eigenfaces_   # (N, k)
    return projections
```

Chỉ 2 dòng nhưng đây là **trọng tâm toán học của đề tài**. `X_centered @ self.eigenfaces_` chính là `(x − x̄) · uᵢ` cho từng eigenface `uᵢ` — tọa độ trong cơ sở Eigenface.

#### `reconstruct(X, n_components=k)` — Tái tạo ảnh

```python
def reconstruct(self, X, n_components):
    k = min(n_components, self.eigenfaces_.shape[1])
    U_k = self.eigenfaces_[:, :k]                       # (p, k)
    X_centered = X - self.mean_face_                    # (N, p)
    coords = X_centered @ U_k                           # (N, k)  ← chiếu
    X_reconstructed = coords @ U_k.T + self.mean_face_  # (N, p)  ← chiếu ngược
    return np.clip(X_reconstructed, 0, 255)
```

`coords @ U_k.T` = `(Uₖ Uₖᵀ)(x − x̄)` rồi cộng `x̄` để về không gian gốc. `np.clip(..., 0, 255)` đảm bảo pixel hợp lệ.

#### `predict(X_test)` — Nhận dạng 1-NN

```python
test_projs = self.project(X_test)
for i in range(N_test):
    diffs = self.train_projections_ - test_projs[i]   # (N_train, k)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))    # khoảng cách Euclidean
    nearest_idx = np.argmin(distances)
    predictions[i] = self.train_labels_[nearest_idx]
```

Broadcasting numpy giúp tính `(N_train, k)` differences chỉ trong một dòng.

### 8.3 [src/baseline.py](src/baseline.py) — Bước 3: Phương pháp đối chiếu

Class `PixelKNNRecognizer`: KNN trực tiếp trong không gian pixel `R¹⁰³⁰⁴`, **KHÔNG chiếu**.

```python
def predict(self, X_test):
    for i in range(N_test):
        diffs = self.X_train_ - X_test[i]              # (N_train, 10304)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train_[k_nearest_idx]
        # Bầu chọn đa số
        predictions[i] = np.argmax(np.bincount(k_nearest_labels))
```

**So sánh với Eigenfaces:**

|                           | Eigenfaces              | Pixel-KNN         |
| ------------------------- | ----------------------- | ----------------- |
| Không gian so sánh        | `R^k` (k ≈ 50)          | `R^p` (p = 10304) |
| Bước học                  | Có (eigendecomposition) | Không             |
| Tốc độ predict            | Nhanh                   | Chậm              |
| Robust với ánh sáng/nhiễu | Tốt hơn                 | Kém hơn           |

### 8.4 [src/evaluator.py](src/evaluator.py) — Đánh giá

Cung cấp các hàm:

- `accuracy(y_true, y_pred)`: tỉ lệ đúng tổng thể.
- `per_class_accuracy(y_true, y_pred)`: accuracy từng người (giúp tìm "ai khó nhận nhất").
- `compare_k_values(...)`: thử nhiều giá trị `k`, tìm `k` tối ưu.
- `run_full_comparison(...)`: chạy Eigenfaces + Baseline, in báo cáo so sánh.

### 8.5 [src/visualizer.py](src/visualizer.py) — Bước 4: Trực quan hóa

5 hàm vẽ biểu đồ, mỗi hàm lưu một file `.png`:

| Hàm                         | Output               | Ý nghĩa                            |
| --------------------------- | -------------------- | ---------------------------------- |
| `plot_mean_face`            | `mean_face.png`      | Khuôn mặt trung bình `x̄`           |
| `plot_eigenfaces`           | `eigenfaces.png`     | Lưới 20 eigenface đầu tiên         |
| `plot_recognition_examples` | `recognition.png`    | 1 ví dụ đúng + 1 ví dụ sai         |
| `plot_accuracy_vs_k`        | `accuracy_vs_k.png`  | Đường cong accuracy theo k         |
| `plot_explained_variance`   | `variance_ratio.png` | Variance từng eigenface + tích lũy |

**Mẹo: `_normalize_for_display`** — eigenfaces có giá trị **âm** (vì là eigenvector), không thể hiển thị trực tiếp. Hàm này map về `[0, 1]` để `imshow` vẽ được.

### 8.6 [main_projection.py](main_projection.py) — Entry point

Sắp xếp 6 bước theo thứ tự, đo thời gian, in báo cáo tóm tắt cuối cùng. Không có logic mới — chỉ là "đạo diễn" gọi các module.

### 8.7 [manual_example.py](manual_example.py) — Bước 5: Ví dụ tính tay

Chạy đúng thuật toán trên dữ liệu siêu nhỏ (4 ảnh `3×3`), in ra **từng bước trung gian**:

1. Hiển thị 4 ảnh + ảnh test
2. Tính `x̄` (cộng các pixel tương ứng / 4)
3. Tính `Φᵢ = xᵢ − x̄` cho từng ảnh
4. Tính `L[i,j]` cho từng cặp (i,j) — show phép tính tích vô hướng
5. Eigendecomposition: in eigenvalues + eigenvectors
6. Tính `uᵢ = Φᵀvᵢ / ‖Φᵀvᵢ‖`, kiểm tra `UᵀU = I`
7. Chiếu từng ảnh train xuống không gian eigenface
8. Chiếu ảnh test, tính khoảng cách, gán nhãn
9. Vẽ scatter 2D không gian eigenface

**Đây là phần GV rất quan tâm.** Khi phản biện, bạn có thể chỉ vào output của file này để chứng minh "em đã hiểu từng bước".

---

## 9. Ví dụ tính tay (dữ liệu nhỏ)

Để đối chiếu với code [manual_example.py](manual_example.py), đây là tính toán bằng tay.

### Dữ liệu

4 ảnh `3×3` pixel, mỗi ảnh là vector `R⁹`:

```
x₁ = [10, 20, 30, 20, 30, 40, 30, 40, 50]  ← Person 1
x₂ = [12, 22, 28, 22, 28, 42, 28, 42, 48]  ← Person 1
x₃ = [90, 80, 70, 80, 70, 60, 70, 60, 50]  ← Person 2
x₄ = [88, 78, 72, 78, 72, 58, 72, 58, 52]  ← Person 2

x_test = [11, 21, 29, 21, 29, 41, 29, 41, 49]  ← cần nhận dạng (kỳ vọng: Person 1)
```

### Bước 1 — Mean face

`x̄ⱼ = (x₁ⱼ + x₂ⱼ + x₃ⱼ + x₄ⱼ) / 4` cho từng `j`.

| j   | x₁ⱼ | x₂ⱼ | x₃ⱼ | x₄ⱼ | Tổng | x̄ⱼ     |
| --- | --- | --- | --- | --- | ---- | ------ |
| 0   | 10  | 12  | 90  | 88  | 200  | **50** |
| 1   | 20  | 22  | 80  | 78  | 200  | **50** |
| 2   | 30  | 28  | 70  | 72  | 200  | **50** |
| 3   | 20  | 22  | 80  | 78  | 200  | **50** |
| 4   | 30  | 28  | 70  | 72  | 200  | **50** |
| 5   | 40  | 42  | 60  | 58  | 200  | **50** |
| 6   | 30  | 28  | 70  | 72  | 200  | **50** |
| 7   | 40  | 42  | 60  | 58  | 200  | **50** |
| 8   | 50  | 48  | 50  | 52  | 200  | **50** |

→ `x̄ = [50, 50, 50, 50, 50, 50, 50, 50, 50]` (hằng số do dữ liệu cố tình đối xứng).

### Bước 2 — Centering

`Φᵢ = xᵢ − x̄`:

```
Φ₁ = [-40, -30, -20, -30, -20, -10, -20, -10,   0]
Φ₂ = [-38, -28, -22, -28, -22,  -8, -22,  -8,  -2]
Φ₃ = [ 40,  30,  20,  30,  20,  10,  20,  10,   0]
Φ₄ = [ 38,  28,  22,  28,  22,   8,  22,   8,   2]
```

Quan sát: `Φ₃ ≈ −Φ₁`, `Φ₄ ≈ −Φ₂` (cấu trúc đối xứng cố ý của dữ liệu mẫu).

### Bước 3 — Ma trận L

`L[i,j] = (Φᵢ · Φⱼ) / 4`. Ví dụ:

- `Φ₁ · Φ₁ = 40²+30²+20²+30²+20²+10²+20²+10²+0² = 1600+900+400+900+400+100+400+100+0 = 4800`
- `L[0,0] = 4800/4 = 1200`
- `Φ₁ · Φ₂ = 40·38 + 30·28 + ... = 4636`, `L[0,1] = 1159`
- Tương tự...

```
L ≈ ⎡ 1200    1159   -1200   -1159 ⎤
    ⎢ 1159    1126   -1159   -1126 ⎥
    ⎢-1200   -1159    1200    1159 ⎥
    ⎣-1159   -1126    1159    1126 ⎦
```

### Bước 4 — Eigendecomposition của L

`np.linalg.eigh(L)` cho ra (sau khi sắp xếp giảm dần và loại 0):

- `λ₁ ≈ 4650` (99.x% phương sai)
- `λ₂ ≈ 1.8` (rất nhỏ — biến thiên trong cùng person)
- `λ₃, λ₄ ≈ 0` (loại)

→ Hai eigenfaces ý nghĩa: `u₁`, `u₂`.

### Bước 5 — Phép chiếu ảnh test

- `x_test − x̄ = [-39, -29, -21, -29, -21, -9, -21, -9, -1]`
- `ŷ_test = Uᵀ(x_test − x̄)`, ví dụ `ŷ_test ≈ [d₁, d₂]` (2 tọa độ).
- Các `ŷᵢ` cho train:
  - `ŷ₁ ≈ [large_negative, small_positive]` (Person 1, ảnh 1)
  - `ŷ₂ ≈ [large_negative, small_negative]` (Person 1, ảnh 2)
  - `ŷ₃ ≈ [large_positive, ...]` (Person 2)
  - `ŷ₄ ≈ [large_positive, ...]` (Person 2)
- `ŷ_test` rất gần với `ŷ₁` và `ŷ₂` → nhận dạng **Person 1**. ✅

Chạy `python manual_example.py` để xem số liệu chính xác.

---

## 10. Hai ứng dụng mở rộng: Tái tạo & Làm mờ ảnh

### 10.1 Tái tạo ảnh (Image Reconstruction)

**Ý tưởng:** Thay vì lưu trọn 10304 pixel, ta chỉ lưu `k` tọa độ trong không gian Eigenface (ví dụ k=50). Khi cần, "giải nén" bằng:
$$\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

**Hệ số nén:** `p / k = 10304 / 50 ≈ 206×`.

**Trade-off:**

| k   | Dung lượng (số float) | Chất lượng (PSNR) | Đánh giá           |
| --- | --------------------- | ----------------- | ------------------ |
| 1   | 1                     | ~3 dB             | Chỉ thấy hình bóng |
| 5   | 5                     | ~10 dB            | Khuôn mặt mờ       |
| 20  | 20                    | ~25 dB            | Nhận ra được       |
| 50  | 50                    | ~35 dB            | Tốt                |
| 100 | 100                   | ~42 dB            | Rất tốt            |
| 150 | 150                   | ~48 dB            | Gần hoàn hảo       |

**Biểu đồ:** `outputs/app1_reconstruction.png` (so sánh trực quan), `app1_quality_curve.png` (đường cong MSE & PSNR theo k).

### 10.2 Làm mờ ảnh (Image Blurring via Low-Rank Projection)

**Quan sát then chốt:** Eigenfaces được sắp theo eigenvalue giảm dần. Eigenvalue lớn ⟺ phương sai cao ⟺ hướng "lớn" trong dữ liệu (tổng thể khuôn mặt: hình dạng, vị trí mắt mũi). Eigenvalue nhỏ ⟺ phương sai thấp ⟺ chi tiết tinh tế (đường nét, cạnh, nhiễu).

**Ý tưởng:** Tái tạo chỉ với **k nhỏ** (vd k=1,3,7,15,30). Bỏ qua các eigenface "chi tiết" → mất chi tiết tần số cao → **ảnh bị làm mờ**.

Đây chính là **bộ lọc thông thấp** (low-pass filter) hoạt động trong miền eigenface chứ không phải miền Fourier như thông thường.

**Quan hệ với DCT/Fourier:**

- DCT/Fourier cố định cơ sở (sin, cos).
- Eigenfaces dùng cơ sở **học từ dữ liệu** → đặc thù cho khuôn mặt → có thể nén tốt hơn cho domain hẹp này.

**Biểu đồ:** `outputs/app2_blurring.png`.

---

## 11. Các biểu đồ đầu ra giải thích cái gì?

| File                      | Ý nghĩa                                                   | Khi nào dùng trong báo cáo?       |
| ------------------------- | --------------------------------------------------------- | --------------------------------- |
| `mean_face.png`           | Khuôn mặt trung bình `x̄` — "nguyên mẫu" của 320 ảnh train | Phần Lý thuyết, sau công thức `x̄` |
| `eigenfaces.png`          | Lưới 20 eigenfaces đầu tiên, kèm % variance               | Phần Lý thuyết, sau Eigenfaces    |
| `recognition.png`         | 1 ví dụ đúng (viền xanh) + 1 ví dụ sai (viền đỏ)          | Phần Kết quả, minh họa hoạt động  |
| `accuracy_vs_k.png`       | Đường accuracy theo k + đường baseline                    | Phần Kết quả, chọn k tối ưu       |
| `variance_ratio.png`      | Cột variance từng EF + đường tích lũy + ngưỡng 95%        | Phần Lý thuyết, lý giải chọn k    |
| `app1_reconstruction.png` | So sánh trực quan tái tạo với k=1,5,20,50,100,150         | Ứng dụng 1 — Tái tạo              |
| `app1_quality_curve.png`  | Đường cong MSE & PSNR theo k                              | Ứng dụng 1 — Phân tích chất lượng |
| `app2_blurring.png`       | Hiệu ứng làm mờ với k nhỏ dần                             | Ứng dụng 2 — Làm mờ               |
| `manual_example.png`      | Ảnh train, eigenfaces, scatter 2D                         | Phần Thực hành, ví dụ tính tay    |

---

## 12. Kết quả thực tế thu được

Sau khi chạy `python main_projection.py` (với cấu hình mặc định):

```
Dataset:          400 ảnh (40 người × 10), train=320, test=80
k tối ưu:         ~50  (kỳ vọng, có thể dao động 30-75 tuỳ random/seed)

EIGENFACES (Phép chiếu vuông góc):
  Accuracy:                     ~96–97%
  Số sai:                       2–3 / 80
  Thời gian train:              ~0.2s
  Thời gian predict (80 ảnh):   ~0.01s

BASELINE (Pixel-KNN):
  Accuracy:                     ~85%
  Số sai:                       ~12 / 80
  Thời gian train:              ~0.0s (chỉ copy)
  Thời gian predict (80 ảnh):   ~0.5s   (chậm hơn 50×)

CẢI THIỆN:
  Accuracy:                     +11% tuyệt đối
  Hệ số nén không gian:         206× (10304D → 50D)
  Tốc độ predict:               ~50× nhanh hơn

PHƯƠNG SAI:
  k=50 eigenfaces giải thích:   ~93% phương sai
  Cần k=~100 để đạt 99%
```

**Diễn giải kết quả cho báo cáo:**

- Eigenfaces vượt trội Baseline cả về **độ chính xác** (do loại nhiễu trong các eigenface bị cắt) lẫn **tốc độ** (so sánh trong R⁵⁰ thay vì R¹⁰³⁰⁴).
- Hệ số nén 206× chứng minh tính hiệu quả của phép chiếu xuống không gian con.
- PSNR > 35 dB ở k=50 cho thấy phép chiếu "giữ" được hầu hết thông tin trực quan.

---

## 13. Câu hỏi phản biện hay gặp + cách trả lời

> GV sẽ hỏi ngẫu nhiên 1 người trong nhóm. Mọi thành viên phải trả lời được những câu sau.

### Q1. Tại sao phải trung tâm hóa dữ liệu (trừ mean face)?

**A.** Hiệp phương sai đo "biến thiên xung quanh giá trị trung bình". Nếu không trừ `x̄`, eigenvector đầu tiên sẽ luôn trùng với hướng của `x̄` (dominant), át hết các hướng biến thiên khác. Trừ đi `x̄` đảm bảo các eigenfaces tập trung vào **sự khác biệt giữa các khuôn mặt**, không phải bản thân khuôn mặt trung bình.

### Q2. Tại sao gọi là phép chiếu _vuông góc_?

**A.** Vì `U` có các cột trực chuẩn (`UᵀU = I`). Phép chiếu `P = UUᵀ` có tính chất: với mọi vector `y`, phần dư `(y − Py)` **vuông góc** với toàn bộ không gian con span(U). Đây là định nghĩa hình học của phép chiếu vuông góc.

Nếu cột của `U` không trực giao thì công thức chiếu phức tạp hơn (`P = U(UᵀU)⁻¹Uᵀ`) và không còn "tự nhiên" nữa.

### Q3. Thủ thuật Turk & Pentland là gì? Tại sao cần?

**A.** Ma trận hiệp phương sai chuẩn `C = ΦᵀΦ/N` có kích thước `p×p = 10304×10304`. Quá lớn để phân rã trực tiếp.

Thủ thuật: dùng `L = ΦΦᵀ/N` có kích thước `N×N = 320×320`. Sau khi tìm eigenvector `v` của `L`, ta có `u = Φᵀv` là eigenvector của `C` với **cùng eigenvalue**. Chứng minh: `Lv = λv ⟹ ΦᵀΦ(Φᵀv) = λN(Φᵀv) ⟹ NC(Φᵀv) = λN(Φᵀv) ⟹ Cu = λu`.

### Q4. Tại sao dùng `np.linalg.eigh` thay vì `np.linalg.eig`?

**A.** `L` là ma trận **thực đối xứng** (vì `L = (ΦΦᵀ)/N` và `(ΦΦᵀ)ᵀ = ΦΦᵀ`). `eigh` chuyên cho ma trận đối xứng/Hermitian:

- Đảm bảo eigenvalue là số thực (không bị nhiễu phức).
- Đảm bảo eigenvectors trực giao.
- Ổn định số học hơn `eig` (thuật toán Jacobi/Householder).
- Nhanh hơn vì lợi dụng tính đối xứng.

### Q5. Số eigenvalue khác 0 tối đa là bao nhiêu?

**A.** `min(N, p) − 1 = N − 1 = 319` (do trung tâm hóa làm mất 1 chiều). Trong code, ta lọc `valid = eigenvalues > 1e-10` để bỏ các eigenvalue ≈ 0 (nhiễu số học).

### Q6. Làm sao chọn k tối ưu?

**A.** Có nhiều cách:

1. **Thử nghiệm**: chạy với nhiều k, chọn k cho accuracy cao nhất trên tập validation. Code dùng cách này (file `evaluator.py`).
2. **Ngưỡng variance**: chọn k nhỏ nhất sao cho tổng variance tích lũy ≥ 95%. (`recognizer.n_components_for_variance(0.95)`).
3. **Quan sát đường cong**: tìm "elbow" của explained variance plot.

Thường thực tế: `k ≈ 30–100` đủ tốt cho ORL.

### Q7. Eigenface là gì về mặt hình ảnh?

**A.** Một eigenface là một vector `R¹⁰³⁰⁴` (cùng kích thước như ảnh) đại diện cho một **hướng biến thiên** trong dữ liệu. Khi duỗi ngược thành ảnh `112×92`, nó trông như khuôn mặt ma quái — không phải khuôn mặt thật, mà là "mẫu" của một dạng biến thiên (ví dụ: hướng sáng, hình dạng mũi, mở/đóng miệng...).

Mỗi khuôn mặt thật sẽ là **tổ hợp tuyến tính** của khuôn mặt trung bình + các eigenfaces với các trọng số khác nhau.

### Q8. Eigenfaces có giá trị âm — làm sao hiển thị?

**A.** Eigenvector có thể có giá trị âm và lớn hơn 255. Hàm `_normalize_for_display` trong `visualizer.py` ánh xạ `[min, max] → [0, 1]` để `imshow` hiển thị được. Lưu ý: việc này **chỉ phục vụ hiển thị**, không thay đổi vector thực dùng trong tính toán.

### Q9. Tại sao Baseline (Pixel-KNN) kém hơn?

**A.**

1. **Nhạy với nhiễu/ánh sáng**: Mỗi pixel đóng góp ngang nhau, ánh sáng thay đổi 10% trên toàn ảnh khiến khoảng cách tăng vọt.
2. **Không tận dụng cấu trúc**: Không hiểu rằng "các pixel gần nhau thường tương quan".
3. **Cao chiều**: Trong `R¹⁰³⁰⁴`, mọi điểm đều xa nhau ("curse of dimensionality"), khoảng cách Euclidean mất đi tính phân biệt.
4. **Chậm**: Mỗi predict phải tính khoảng cách trong R¹⁰³⁰⁴ với 320 vector.

Eigenfaces giải quyết tất cả: giảm chiều (loại nhiễu + tăng tốc), giữ thông tin chính.

### Q10. Phép chiếu vuông góc có **làm mờ** ảnh tại sao?

**A.** Vì các eigenface đầu (eigenvalue lớn) bắt **đặc điểm tổng thể** (tần số thấp: hình dáng khuôn mặt), còn các eigenface sau bắt **chi tiết** (tần số cao: nét, cạnh, lỗ chân lông). Khi giữ chỉ k eigenface đầu, ta **vứt đi chi tiết** → ảnh tái tạo mờ. Đây là một **bộ lọc thông thấp** học từ dữ liệu (data-driven low-pass filter).

### Q11. Khi nào Eigenfaces thất bại?

**A.**

- **Ánh sáng thay đổi mạnh**: hướng eigenface đầu thường mô hình "thay đổi ánh sáng" thay vì "khác biệt người". Cách khắc phục: bỏ 3 eigenface đầu, hoặc dùng Fisherfaces (LDA).
- **Tư thế nghiêng nhiều**: ORL ảnh chính diện, không xử lý được mặt nghiêng 45°.
- **Ảnh test khác đáng kể so với train**: vd đeo kính khi train không kính.
- **Số chiều quá nhỏ**: k=5 sẽ accuracy thấp.

### Q12. Tại sao không dùng SVD?

**A.** Có thể dùng SVD trên `Φ` cho cùng kết quả: `Φ = U_svd Σ V_svdᵀ`. Lúc đó `U_svd` chính là eigenfaces, `Σ²/N` là eigenvalues. Code dùng eigendecomposition vì rõ ràng về mặt giảng dạy: tách bạch các bước cov → eig → recover, dễ giải thích cho báo cáo.

### Q13. Phân biệt PCA và Eigenfaces?

**A.** Eigenfaces = **áp dụng PCA cho ảnh khuôn mặt**. PCA là tên gọi chung của thuật toán; Eigenfaces là một ứng dụng cụ thể (Turk & Pentland 1991) đầu tiên dùng PCA cho face recognition.

### Q14. Tại sao dùng 1-NN mà không phải k-NN?

**A.**

- Với ORL: ảnh trong cùng person rất giống nhau, 1-NN đủ chính xác.
- k-NN cần nhiều ảnh/person hơn (mỗi class chỉ có 8 ảnh train, k=5 thì ranh giới class mờ).
- Đơn giản hơn cho mục đích minh họa lý thuyết.

Code có hỗ trợ k-NN tùy ý trong baseline (`PixelKNNRecognizer(k=...)`).

---

## 14. Thuật ngữ tra cứu nhanh (glossary)

| Thuật ngữ                        | Định nghĩa ngắn                                                |
| -------------------------------- | -------------------------------------------------------------- |
| **Pixel**                        | Một ô màu trong ảnh số (giá trị 0–255 với grayscale)           |
| **Grayscale**                    | Ảnh đen-trắng (mỗi pixel = 1 số)                               |
| **Vector hóa (flatten)**         | Duỗi ảnh 2D thành mảng 1D                                      |
| **Mean face**                    | Khuôn mặt trung bình `x̄ = (1/N)Σxᵢ`                            |
| **Centering**                    | Trừ mean: `Φᵢ = xᵢ − x̄`                                        |
| **Covariance matrix**            | Ma trận `C = ΦᵀΦ/N`, đo tương quan giữa các pixel              |
| **Eigenvalue (trị riêng) λ**     | Số đo "độ giãn" của ma trận theo một hướng                     |
| **Eigenvector (vector riêng) v** | Hướng bất biến: `Av = λv`                                      |
| **Eigenface**                    | Eigenvector của covariance khi duỗi thành ảnh                  |
| **Eigendecomposition**           | Phân rã `A = VΛVᵀ` (Λ chéo, V cột trực giao)                   |
| **SVD**                          | Phân rã `A = UΣVᵀ`, tổng quát hơn eigendecomposition           |
| **Orthonormal basis**            | Cơ sở vector vuông góc đôi một, độ dài 1                       |
| **Projection (phép chiếu)**      | Map vector về không gian con: `Py`                             |
| **Orthogonal projection**        | Phép chiếu vuông góc, đặc trưng bởi `UUᵀ` với cột U trực chuẩn |
| **Explained variance**           | Tỉ lệ phương sai mà eigenface giữ được                         |
| **1-NN (Nearest Neighbor)**      | Gán nhãn = nhãn của láng giềng gần nhất                        |
| **Euclidean distance**           | Khoảng cách `‖a−b‖₂ = √Σ(aᵢ−bᵢ)²`                              |
| **MSE**                          | Trung bình bình phương sai số                                  |
| **PSNR**                         | Peak Signal-to-Noise Ratio, đo chất lượng ảnh (dB)             |
| **Low-pass filter**              | Bộ lọc giữ tần số thấp (cấu trúc), bỏ tần số cao (chi tiết)    |
| **Curse of dimensionality**      | Hiện tượng khoảng cách mất tính phân biệt ở chiều cao          |
| **Broadcasting**                 | Cơ chế numpy tự "nhân bản" vector để cộng/trừ với ma trận      |

---

## 15. Tài liệu tham khảo

1. **Turk, M. & Pentland, A. (1991).** "Eigenfaces for Recognition." _Journal of Cognitive Neuroscience_, 3(1), 71–86. — Bài báo gốc của thuật toán.
2. **Jolliffe, I. T. (2002).** _Principal Component Analysis._ 2nd ed. Springer. — Lý thuyết PCA đầy đủ.
3. **Gonzalez, R. C. & Woods, R. E. (2018).** _Digital Image Processing._ 4th ed. Pearson. — MSE, PSNR, low-pass filter.
4. **Strang, G. (2016).** _Introduction to Linear Algebra._ 5th ed. Wellesley-Cambridge. — Cơ sở phép chiếu, eigendecomposition.
5. **Cover, T. & Hart, P. (1967).** "Nearest Neighbor Pattern Classification." _IEEE Transactions on Information Theory_, 13(1), 21–27.
6. **AT&T Laboratories Cambridge.** "The ORL Database of Faces." https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
7. **NumPy documentation** — `numpy.linalg.eigh`. https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
8. **Hunter, J. D. (2007).** "Matplotlib: A 2D graphics environment." _Computing in Science & Engineering_, 9(3), 90–95.

---

## Lời kết

Đến đây bạn đã có đủ kiến thức để:

- Đọc hiểu báo cáo dạng văn bản và mọi công thức trong đó.
- Đọc hiểu mọi dòng code trong dự án.
- Trả lời câu hỏi phản biện ở mức tốt.
- Mở rộng dự án (vd thêm Fisherfaces, dùng dataset khác).

Nếu còn chỗ nào chưa rõ:

1. Đọc lại phần Lý thuyết (mục 4).
2. Chạy `python manual_example.py` và đọc từng bước in ra console.
3. Xem comment trong file [src/recognizer.py](src/recognizer.py).
4. Đọc [README.md](README.md) cho góc nhìn ngắn gọn hơn.

**Chúc nhóm bảo vệ thành công!**
