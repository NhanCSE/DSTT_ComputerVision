# CÁC LỆNH CHÍNH CỦA DỰ ÁN

> Bản rút gọn của [COMMANDS.md](COMMANDS.md) — chỉ giữ **những lệnh cốt lõi** thực sự xuất hiện trong source code và cần đưa vào báo cáo.
>
> Phạm vi: dự án **Eigenfaces — Nhận dạng khuôn mặt bằng phép chiếu vuông góc** (ORL Faces, 40 người × 10 ảnh).

---

## 1. LỆNH CHẠY CHƯƠNG TRÌNH (Shell)

| STT | Lệnh | Vai trò trong dự án |
|---|---|---|
| 1 | `pip install -r requirements.txt` | Cài thư viện: `numpy`, `Pillow`, `matplotlib`, `streamlit`. |
| 2 | `python main_projection.py` | Chạy toàn bộ pipeline **Bước 1 → Bước 4**: tải dữ liệu → huấn luyện Eigenfaces → so sánh với Baseline → sinh biểu đồ vào `outputs/`. |
| 3 | `python manual_example.py` | Chạy ví dụ tính tay (Bước 5) trên 4 ảnh 3×3 — in từng phép tính chi tiết. |
| 4 | `streamlit run web/streamlit_app.py` | Khởi động ứng dụng web demo (giao diện trên trình duyệt). |

---

## 2. LỆNH NUMPY — LÕI THUẬT TOÁN EIGENFACES

Đây là các lệnh **trực tiếp hiện thực phép chiếu vuông góc** trong [src/recognizer.py](src/recognizer.py).

| STT | Lệnh | Ý nghĩa | Vị trí dùng |
|---|---|---|---|
| 1 | `np.mean(X, axis=0)` | Tính **khuôn mặt trung bình** $\bar{x} = \frac{1}{N}\sum x_i$ theo từng pixel. | [recognizer.py:90](src/recognizer.py#L90) |
| 2 | `X - mean_face` (broadcasting) | **Trung tâm hóa**: $\Phi_i = x_i - \bar{x}$ — tự động trừ `mean_face` khỏi mọi hàng của `X`. | [recognizer.py:98](src/recognizer.py#L98) |
| 3 | `@` / `np.matmul` | Nhân ma trận. Dùng cho $L = \frac{1}{N}\Phi\Phi^T$, $U = \Phi^T V$, và phép chiếu $(X - \bar{x})\,U$. | [recognizer.py:116,149,208](src/recognizer.py#L116) |
| 4 | `.T` | Chuyển vị ma trận — đổi hàng thành cột. Dùng trong $\Phi^T$, $U^T$. | [recognizer.py:116,149](src/recognizer.py#L116) |
| 5 | `np.linalg.eigh(L)` | **Phân rã trị riêng** của ma trận thực đối xứng. Trả về `(eigenvalues, eigenvectors)`. Ổn định số học hơn `np.linalg.eig`. | [recognizer.py:128](src/recognizer.py#L128) |
| 6 | `np.argsort(arr)[::-1]` | Sắp xếp eigenvalues giảm dần để chọn k eigenfaces "quan trọng nhất". | [recognizer.py:133](src/recognizer.py#L133) |
| 7 | `np.linalg.norm(U, axis=0)` | Tính độ dài từng cột để **chuẩn hóa eigenfaces** về `‖u_i‖ = 1` (orthonormal basis). | [recognizer.py:152](src/recognizer.py#L152) |
| 8 | `np.sqrt(np.sum(diffs**2, axis=1))` | Tính **khoảng cách Euclidean** $\|\hat{y} - \hat{y}_i\|_2$ trong không gian eigenface. | [recognizer.py:288](src/recognizer.py#L288) |
| 9 | `np.argmin(distances)` | Tìm láng giềng gần nhất (**1-NN**) — chọn ảnh train có khoảng cách nhỏ nhất. | [recognizer.py:292](src/recognizer.py#L292) |
| 10 | `np.clip(X, 0, 255)` | Cắt giá trị pixel sau khi tái tạo về dải hợp lệ `[0, 255]`. | [recognizer.py:246](src/recognizer.py#L246) |
| 11 | `np.cumsum(ratio)` / `np.searchsorted(cumvar, 0.95)` | Tính phương sai tích lũy và tìm `k` tối thiểu đạt 95% phương sai. | [recognizer.py:315,323](src/recognizer.py#L315) |
| 12 | `.reshape(112, 92)` | Chuyển vector 1D `(10304,)` → ảnh 2D `(112, 92)` để hiển thị. | [visualizer.py:40](src/visualizer.py#L40) |
| 13 | `.flatten()` | **Vector hóa ảnh**: chuyển ảnh 2D → vector 1D — bước đầu tiên trước khi áp dụng đại số tuyến tính. | [dataloader.py:189](src/dataloader.py#L189) |
| 14 | `np.mean(y_true == y_pred)` | Tính **accuracy** — tận dụng broadcasting và quy ước `True=1, False=0`. | [evaluator.py:37](src/evaluator.py#L37) |

---

## 3. LỆNH PILLOW — XỬ LÝ ẢNH ĐẦU VÀO

Dùng trong [src/dataloader.py](src/dataloader.py) để đọc file ảnh `.pgm` của ORL dataset.

| STT | Lệnh | Ý nghĩa |
|---|---|---|
| 1 | `Image.open(path)` | Mở file ảnh (`.pgm`, `.jpg`, `.png`). |
| 2 | `img.convert("L")` | Chuyển ảnh sang **grayscale** 8-bit (mỗi pixel 0–255). |
| 3 | `img.resize((92, 112))` | Đổi kích thước về chuẩn ORL — đảm bảo mọi vector cùng độ dài 10304. |
| 4 | `np.array(img, dtype=np.float64)` | Chuyển ảnh PIL → mảng NumPy `float64` để tham gia đại số tuyến tính. |

---

## 4. LỆNH MATPLOTLIB — VẼ BIỂU ĐỒ BÁO CÁO

Dùng trong [src/visualizer.py](src/visualizer.py) để sinh 5 hình lưu vào `outputs/`.

| STT | Lệnh | Vai trò |
|---|---|---|
| 1 | `plt.subplots(figsize=...)` | Tạo figure + axes — khung của mọi biểu đồ. |
| 2 | `ax.imshow(img, cmap="gray", vmin=0, vmax=255)` | Hiển thị **khuôn mặt trung bình** và **eigenfaces** dưới dạng ảnh xám. |
| 3 | `ax.plot(k_values, accuracies, marker="o")` | Vẽ đường **Accuracy theo k** (Hình 4). |
| 4 | `ax.bar(indices, evr)` | Vẽ cột **phương sai từng eigenface** (Hình 5). |
| 5 | `ax.axhline(y=95, linestyle="--")` | Vẽ ngưỡng 95% phương sai. |
| 6 | `ax.twinx()` | Tạo trục y thứ hai cho **phương sai tích lũy** (biểu đồ kép). |
| 7 | `ax.annotate("k tốt nhất", xy=..., arrowprops=...)` | Chú thích có mũi tên chỉ vào điểm k tối ưu. |
| 8 | `fig.savefig(path, dpi=150, bbox_inches="tight")` | Lưu hình ra PNG với độ phân giải cao cho báo cáo. |
| 9 | `plt.close(fig)` | Đóng figure giải phóng bộ nhớ (chạy nhiều biểu đồ liên tiếp). |

---

## 5. LỆNH STREAMLIT — GIAO DIỆN WEB DEMO

Dùng trong [web/streamlit_app.py](web/streamlit_app.py) — giao diện cho người dùng cuối.

| STT | Lệnh | Vai trò |
|---|---|---|
| 1 | `st.set_page_config(page_title=..., layout="wide")` | Cấu hình tab trình duyệt và layout. |
| 2 | `st.title` / `st.markdown` | Tiêu đề và nội dung markdown. |
| 3 | `st.columns(2)` | Chia layout 2 cột song song (so sánh ảnh trước–sau). |
| 4 | `st.tabs([...])` | Tab chuyển đổi giữa các bước thuật toán. |
| 5 | `st.slider("k", 5, 150, 50)` | Thanh trượt chọn số eigenfaces. |
| 6 | `st.file_uploader("Upload", type=["jpg","png"])` | Cho phép upload ảnh để nhận dạng. |
| 7 | `st.image(arr, caption=...)` | Hiển thị ảnh kết quả (mean face, eigenfaces, ảnh tái tạo). |
| 8 | `st.metric("Accuracy", "...")` | Hiển thị số liệu nổi bật. |
| 9 | `@st.cache_data` / `@st.cache_resource` | Cache dataset và mô hình đã huấn luyện — tránh chạy lại mỗi lần re-run. |
| 10 | `st.session_state` | Lưu trạng thái mô hình giữa các lần re-run. |

---

## 6. ĐỐI CHIẾU CÔNG THỨC TOÁN ↔ LỆNH NUMPY

Bảng này là **mối nối giữa giải tích/đại số tuyến tính của báo cáo và code** — phần quan trọng nhất khi trình bày.

| STT | Công thức toán | Lệnh NumPy | Ý nghĩa |
|---|---|---|---|
| 1 | $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ | `np.mean(X_train, axis=0)` | Khuôn mặt trung bình. |
| 2 | $\Phi_i = x_i - \bar{x}$ | `Phi = X_train - mean_face` | Trung tâm hóa (broadcasting). |
| 3 | $L = \frac{1}{N}\Phi\Phi^T$ | `L = (Phi @ Phi.T) / N` | Ma trận hiệp phương sai thay thế (Turk–Pentland trick). |
| 4 | $Lv = \lambda v$ | `λ, V = np.linalg.eigh(L)` | Phân rã trị riêng. |
| 5 | $u_i = \dfrac{\Phi^T v_i}{\|\Phi^T v_i\|}$ | `U = (Phi.T @ V) / norms` | Khôi phục eigenfaces và chuẩn hóa. |
| 6 | $\hat{y} = U^T(x - \bar{x})$ | `proj = (X - mean_face) @ U` | **Phép chiếu vuông góc** — trọng tâm của bài. |
| 7 | $\hat{x} = U_k U_k^T(x - \bar{x}) + \bar{x}$ | `coords @ U_k.T + mean_face` | Tái tạo ảnh (chiếu rồi chiếu ngược). |
| 8 | $d = \|\hat{y} - \hat{y}_i\|_2$ | `np.sqrt(np.sum((p - q)**2))` | Khoảng cách Euclidean. |
| 9 | $\arg\min_i d_i$ | `np.argmin(distances)` | Chọn láng giềng gần nhất (1-NN). |
| 10 | $U^T U = I$ | `U.T @ U` (≈ $I$) | Kiểm tra hệ cơ sở **trực chuẩn** (orthonormal). |

---

## 7. QUY TRÌNH CHẠY (gợi ý cho người chấm)

```bash
# Bước A: Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# Bước B: Chạy toàn bộ pipeline — sinh báo cáo + biểu đồ
python main_projection.py

# Bước C: Xem ví dụ tính tay (ảnh 3×3, in chi tiết từng phép tính)
python manual_example.py

# Bước D: Mở giao diện web demo
streamlit run web/streamlit_app.py
```

**Sản phẩm sau khi chạy `main_projection.py`:**
- Báo cáo so sánh Eigenfaces vs Baseline (in ra terminal).
- 5 biểu đồ `.png` lưu trong [outputs/](outputs/): `mean_face.png`, `eigenfaces.png`, `recognition.png`, `accuracy_vs_k.png`, `variance_ratio.png`.

---

## 8. GHI CHÚ NHANH

- **Vector hóa**: ảnh 2D `(112, 92)` → vector 1D `(10304,)` để áp dụng đại số tuyến tính.
- **Thủ thuật Turk–Pentland**: thay vì phân rã ma trận `(10304, 10304)`, ta phân rã `L = ΦΦᵀ/N` có shape `(N, N)` — nhỏ hơn rất nhiều.
- **Phép chiếu vuông góc** = nhân $U^T$ với $(x - \bar{x})$ → "ép" ảnh từ 10304 chiều xuống `k` chiều (vd: k = 50).
- **1-NN trên không gian eigenface** vừa nhanh hơn (so 50 chiều thay vì 10304) vừa chính xác hơn (lọc bỏ nhiễu).
- **`np.linalg.eigh` vs `np.linalg.eig`**: `eigh` dành riêng cho ma trận đối xứng — ổn định số học hơn, đảm bảo trị riêng là số thực.
