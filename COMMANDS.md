# DANH SÁCH CÁC LỆNH ĐÃ SỬ DỤNG TRONG DỰ ÁN

> Tổng hợp tất cả các lệnh (shell, NumPy, Pillow, Matplotlib, Streamlit, Python chuẩn) được sử dụng trong dự án **Eigenfaces — Nhận dạng khuôn mặt bằng phép chiếu vuông góc**.
>
> Mục đích: giúp người mới đọc code có thể tra cứu nhanh từng lệnh mà không cần mở tài liệu chính thức.

---

## 1. LỆNH SHELL / TERMINAL (chạy ngoài Python)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `pip install` | `pip install -r requirements.txt` | Cài tất cả thư viện cần thiết (numpy, matplotlib, Pillow, opencv-python, streamlit) ghi trong file `requirements.txt`. |
| 2 | `python` | `python main_projection.py` | Chạy pipeline đầy đủ (Bước 1 → 4): tải dữ liệu, huấn luyện, so sánh, vẽ biểu đồ. |
| 3 | `python` | `python manual_example.py` | Chạy ví dụ tính tay (Bước 5) trên 4 ảnh 3×3 pixel — in từng phép tính chi tiết. |
| 4 | `python` (module) | `python src/dataloader.py` | Chạy thử riêng module đọc dataset để kiểm tra. |
| 5 | `python` (module) | `python src/recognizer.py` | Chạy thử riêng class `OrthogonalFaceRecognizer` (huấn luyện + dự đoán nhanh). |
| 6 | `python` (module) | `python src/evaluator.py` | Chạy thử module đánh giá: tìm k tối ưu, so sánh Eigenfaces vs Baseline. |
| 7 | `python` (module) | `python src/visualizer.py` | Chạy thử module vẽ tất cả 5 biểu đồ. |
| 8 | `streamlit run` | `streamlit run web/streamlit_app.py` | Khởi động ứng dụng web demo Eigenfaces (giao diện trên trình duyệt). |

---

## 2. LỆNH PYTHON CHUẨN (built-in & thư viện chuẩn)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `import` | `import numpy as np` | Nạp thư viện và đặt bí danh (alias) ngắn để dùng. |
| 2 | `from ... import` | `from PIL import Image` | Nạp một tên cụ thể (class/hàm) từ một module. |
| 3 | `os.path.join` | `os.path.join("data", "orl_faces")` | Ghép đường dẫn theo đúng hệ điều hành (Windows dùng `\`, Linux dùng `/`). |
| 4 | `os.path.isdir` | `os.path.isdir(path)` | Kiểm tra `path` có phải là thư mục đang tồn tại không. |
| 5 | `os.path.exists` | `os.path.exists(path)` | Kiểm tra file/thư mục có tồn tại không. |
| 6 | `os.makedirs` | `os.makedirs(dir, exist_ok=True)` | Tạo thư mục (kể cả các thư mục cha); `exist_ok=True` để không lỗi nếu đã có. |
| 7 | `os.listdir` | `os.listdir(path)` | Liệt kê tên các file/thư mục con trong `path`. |
| 8 | `os.remove` | `os.remove(file)` | Xoá một file. |
| 9 | `os.rename` | `os.rename(src, dst)` | Đổi tên/di chuyển file hoặc thư mục. |
| 10 | `shutil.rmtree` | `shutil.rmtree(dir)` | Xoá đệ quy toàn bộ thư mục (kể cả file con). |
| 11 | `urllib.request.urlretrieve` | `urllib.request.urlretrieve(url, path)` | Tải file từ Internet về máy. |
| 12 | `zipfile.ZipFile` | `with zipfile.ZipFile(zip, "r") as zf:` | Mở file `.zip` để đọc. |
| 13 | `.extractall` | `zf.extractall(dir)` | Giải nén toàn bộ nội dung file zip vào `dir`. |
| 14 | `time.perf_counter` | `t0 = time.perf_counter()` | Lấy thời điểm hiện tại (giây, chính xác cao) — dùng để đo thời gian. |
| 15 | `sys.path.insert` | `sys.path.insert(0, root)` | Thêm thư mục vào danh sách tìm kiếm module để `import` được file ở đó. |
| 16 | `pathlib.Path` | `Path(__file__).resolve().parents[1]` | Lấy đường dẫn (Path object), `parents[1]` = đi lên 1 cấp thư mục. |

---

## 3. LỆNH NUMPY (đại số tuyến tính — lõi của thuật toán)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `np.array` | `np.array([1, 2, 3], dtype=float)` | Tạo mảng NumPy (vector hoặc ma trận) từ list, với kiểu dữ liệu chỉ định. |
| 2 | `np.zeros` | `np.zeros(N, dtype=np.int32)` | Tạo mảng toàn số 0, dùng để khởi tạo trước khi điền giá trị. |
| 3 | `np.mean` | `np.mean(X, axis=0)` | Tính trung bình. `axis=0` → trung bình theo cột (ra **khuôn mặt trung bình** x̄). |
| 4 | `np.sum` | `np.sum(arr, axis=1)` | Tổng các phần tử. `axis=1` → tổng theo hàng. |
| 5 | `np.sqrt` | `np.sqrt(x)` | Căn bậc hai (dùng tính khoảng cách Euclidean). |
| 6 | `np.argmin` | `np.argmin(distances)` | Trả về chỉ số (index) của phần tử **nhỏ nhất** — dùng để tìm láng giềng gần nhất. |
| 7 | `np.argmax` | `np.argmax(votes)` | Trả về chỉ số của phần tử **lớn nhất** (vd: lớp được vote nhiều nhất). |
| 8 | `np.argsort` | `np.argsort(arr)[::-1]` | Trả về chỉ số sắp xếp tăng dần; `[::-1]` để đảo ngược (giảm dần). |
| 9 | `np.unique` | `np.unique(y_train)` | Trả về các giá trị duy nhất (không trùng) — dùng đếm số người trong dataset. |
| 10 | `np.bincount` | `np.bincount(labels)` | Đếm số lần xuất hiện của từng nhãn (chỉ với số nguyên không âm). |
| 11 | `np.where` | `np.where(condition)[0]` | Trả về chỉ số các phần tử thoả điều kiện. |
| 12 | `np.cumsum` | `np.cumsum(ratios)` | Tổng tích lũy — dùng tính phương sai tích lũy. |
| 13 | `np.searchsorted` | `np.searchsorted(cumvar, 0.95)` | Tìm vị trí chèn để giữ thứ tự — dùng tìm k đạt 95% phương sai. |
| 14 | `np.arange` | `np.arange(1, n+1)` | Tạo dãy số `[1, 2, ..., n]`. |
| 15 | `np.linalg.eigh` | `λ, V = np.linalg.eigh(L)` | **Phân rã trị riêng** của ma trận thực **đối xứng**. Trả về eigenvalues (tăng dần) và eigenvectors (mỗi cột). |
| 16 | `np.linalg.norm` | `np.linalg.norm(u, axis=0)` | Tính độ dài (chuẩn) của vector. `axis=0` → norm theo từng cột. |
| 17 | `np.dot` | `np.dot(a, b)` | Tích vô hướng giữa hai vector (hoặc nhân ma trận). |
| 18 | `@` (matmul) | `A @ B` | Toán tử nhân ma trận (tương đương `np.matmul`). |
| 19 | `.T` | `Phi.T` | Chuyển vị (transpose) — đổi hàng thành cột. |
| 20 | `.reshape` | `vec.reshape(112, 92)` | Đổi shape mảng (không sao chép). Dùng đổi vector 1D → ảnh 2D. |
| 21 | `.flatten` | `arr.flatten()` | Duỗi mảng nhiều chiều thành vector 1D. |
| 22 | `.shape` | `X.shape` | Thuộc tính trả về kích thước (vd: `(400, 10304)`). |
| 23 | `.ndim` | `X.ndim` | Số chiều của mảng (1, 2, 3, ...). |
| 24 | `.copy` | `X.copy()` | Sao chép mảng (tránh tham chiếu chung). |
| 25 | `.min` / `.max` | `arr.min()`, `arr.max()` | Giá trị nhỏ nhất / lớn nhất của mảng. |
| 26 | `np.asarray` | `np.asarray(y)` | Ép kiểu sang ndarray (không sao chép nếu đã đúng). |
| 27 | `np.array2string` | `np.array2string(M, separator=", ")` | Định dạng mảng thành chuỗi đẹp khi in. |
| 28 | `np.round` | `np.round(arr, 3)` | Làm tròn đến `n` chữ số sau dấu phẩy. |
| 29 | `np.set_printoptions` | `np.set_printoptions(precision=4)` | Cấu hình cách NumPy in mảng (làm tròn, độ rộng dòng). |
| 30 | Broadcasting | `X - mean_face` | Tự động "phát thanh" mean_face để trừ cho từng hàng của X. |
| 31 | Boolean indexing | `X_train[mask]` | Chỉ lấy các hàng mà `mask=True` (vd: lọc ảnh của một người). |

---

## 4. LỆNH PILLOW (xử lý ảnh)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `Image.open` | `Image.open(path)` | Mở file ảnh (jpg, png, pgm, …). |
| 2 | `.convert("L")` | `img.convert("L")` | Chuyển ảnh sang **grayscale** (1 kênh, mỗi pixel 0–255). |
| 3 | `.resize` | `img.resize((W, H))` | Đổi kích thước ảnh về (rộng, cao). |
| 4 | `Image.open(BytesIO(b))` | `Image.open(io.BytesIO(file_bytes))` | Mở ảnh từ chuỗi byte trong bộ nhớ (dùng cho file upload). |

---

## 5. LỆNH MATPLOTLIB (vẽ biểu đồ)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `plt.subplots` | `fig, ax = plt.subplots(figsize=(7, 4))` | Tạo figure + lưới axes. |
| 2 | `ax.imshow` | `ax.imshow(img, cmap="gray", vmin=0, vmax=255)` | Hiển thị ảnh 2D dưới dạng heatmap. `cmap="gray"` = thang xám. |
| 3 | `ax.plot` | `ax.plot(x, y, marker="o", color="#3498db")` | Vẽ đường line (có thể có marker). |
| 4 | `ax.bar` | `ax.bar(indices, values, color="#3498db")` | Vẽ biểu đồ cột. |
| 5 | `ax.scatter` | `ax.scatter(x, y, c="red", s=80)` | Vẽ scatter plot (chấm điểm). |
| 6 | `ax.axhline` | `ax.axhline(y=95, linestyle="--")` | Vẽ đường ngang ngang qua trục y. |
| 7 | `ax.axvline` | `ax.axvline(x=k, linestyle="--")` | Vẽ đường dọc qua trục x. |
| 8 | `ax.annotate` | `ax.annotate("text", xy=..., xytext=..., arrowprops=...)` | Thêm chú thích có mũi tên chỉ vào một điểm. |
| 9 | `ax.text` | `ax.text(x, y, "abc")` | Đặt một chuỗi văn bản tại toạ độ (x, y). |
| 10 | `ax.set_title` | `ax.set_title("Tiêu đề")` | Đặt tiêu đề cho subplot. |
| 11 | `ax.set_xlabel` / `set_ylabel` | `ax.set_xlabel("k")` | Đặt nhãn trục x / y. |
| 12 | `ax.set_xticks` / `set_yticks` | `ax.set_xticks([5,10,20])` | Đặt mốc giá trị trên trục. |
| 13 | `ax.axis("off")` | `ax.axis("off")` | Ẩn cả khung trục và nhãn (dùng khi hiển thị ảnh). |
| 14 | `ax.legend` | `ax.legend(fontsize=10)` | Hiển thị chú thích (legend) các đường/cột. |
| 15 | `ax.grid` | `ax.grid(True, alpha=0.3)` | Bật/tắt lưới nền, `alpha` = độ trong suốt. |
| 16 | `ax.twinx` | `ax2 = ax1.twinx()` | Tạo trục y thứ hai bên phải — dùng vẽ 2 đại lượng khác đơn vị. |
| 17 | `fig.suptitle` | `fig.suptitle("Tiêu đề lớn")` | Tiêu đề chung cho cả figure. |
| 18 | `fig.tight_layout` | `fig.tight_layout()` | Tự căn chỉnh khoảng trắng giữa các subplot. |
| 19 | `fig.savefig` | `fig.savefig(path, dpi=150, bbox_inches="tight")` | Lưu figure ra file ảnh (PNG, PDF, …). |
| 20 | `plt.close` | `plt.close(fig)` | Đóng figure để giải phóng bộ nhớ. |
| 21 | `plt.show` | `plt.show()` | Hiển thị cửa sổ biểu đồ (chỉ dùng khi chạy interactive). |

---

## 6. LỆNH STREAMLIT (xây dựng web demo)

| STT | Lệnh | Cú pháp | Ý nghĩa |
|---|---|---|---|
| 1 | `st.set_page_config` | `st.set_page_config(page_title="...", layout="wide")` | Cấu hình trang (tiêu đề tab trình duyệt, layout). |
| 2 | `st.title` | `st.title("Tiêu đề")` | Tiêu đề lớn nhất trên trang. |
| 3 | `st.header` / `st.markdown` | `st.markdown("## H2")` | Hiển thị markdown (hỗ trợ heading, bold, ảnh, LaTeX). |
| 4 | `st.write` | `st.write(value)` | In gần như bất cứ thứ gì (text, dataframe, biểu đồ). |
| 5 | `st.caption` | `st.caption("dòng phụ")` | Văn bản nhỏ màu xám (chú thích). |
| 6 | `st.image` | `st.image(arr, caption="...", width=240)` | Hiển thị ảnh (NumPy array hoặc đường dẫn). |
| 7 | `st.columns` | `col1, col2 = st.columns(2)` | Chia layout thành các cột song song. |
| 8 | `st.tabs` | `tab1, tab2 = st.tabs(["A", "B"])` | Tạo các tab chuyển đổi nội dung. |
| 9 | `st.checkbox` | `st.checkbox("Label", value=False)` | Ô tick chọn True/False. |
| 10 | `st.slider` | `st.slider("Label", 5, 150, 50, step=5)` | Thanh trượt chọn giá trị số. |
| 11 | `st.selectbox` | `st.selectbox("Label", options=[...])` | Hộp chọn 1 trong danh sách. |
| 12 | `st.file_uploader` | `st.file_uploader("Upload", type=["jpg","png"])` | Cho người dùng upload file. |
| 13 | `st.button` | `st.button("Chạy")` | Nút bấm; trả về `True` khi vừa bấm. |
| 14 | `st.metric` | `st.metric("Label", "value")` | Khối hiển thị số liệu nổi bật. |
| 15 | `st.progress` | `progress = st.progress(0)` | Thanh tiến trình, cập nhật bằng `.progress(0.5)`. |
| 16 | `st.spinner` | `with st.spinner("Loading..."):` | Hiển thị spinner xoay trong lúc xử lý. |
| 17 | `st.success` / `info` / `warning` | `st.success("OK")` | Hộp thông báo xanh/xanh nhạt/vàng. |
| 18 | `st.code` | `st.code("print('hi')", language="python")` | Hiển thị khối code có highlight. |
| 19 | `st.expander` | `with st.expander("Xem thêm"):` | Khối có thể mở/đóng. |
| 20 | `st.divider` | `st.divider()` | Đường kẻ ngang phân cách. |
| 21 | `st.dataframe` | `st.dataframe(data, use_container_width=True)` | Hiển thị bảng dữ liệu có thể cuộn. |
| 22 | `st.line_chart` | `st.line_chart(data, x="k", y="accuracy")` | Vẽ nhanh đường line từ dict/dataframe. |
| 23 | `st.empty` | `box = st.empty()` | Tạo "slot" rỗng để cập nhật nội dung sau. |
| 24 | `st.session_state` | `st.session_state.optimal_k = 50` | Lưu biến giữa các lần re-run của Streamlit. |
| 25 | `@st.cache_data` | `@st.cache_data` | Cache kết quả hàm trả về dữ liệu (vd: load dataset). |
| 26 | `@st.cache_resource` | `@st.cache_resource` | Cache đối tượng nặng (vd: model đã huấn luyện). |

---

## 7. CÔNG THỨC TOÁN ↔ LỆNH NUMPY (đối chiếu nhanh)

| STT | Công thức toán | Lệnh NumPy tương ứng | Ý nghĩa |
|---|---|---|---|
| 1 | $\bar{x} = \frac{1}{N}\sum x_i$ | `np.mean(X_train, axis=0)` | Khuôn mặt trung bình. |
| 2 | $\Phi_i = x_i - \bar{x}$ | `Phi = X_train - mean_face` | Trung tâm hóa (broadcasting). |
| 3 | $L = \frac{1}{N}\Phi\Phi^T$ | `L = (Phi @ Phi.T) / N` | Ma trận hiệp phương sai thay thế. |
| 4 | $L v = \lambda v$ | `λ, V = np.linalg.eigh(L)` | Phân rã trị riêng. |
| 5 | $u_i = \frac{\Phi^T v_i}{\|\Phi^T v_i\|}$ | `U = (Phi.T @ V) / norms` | Khôi phục eigenfaces (chuẩn hóa). |
| 6 | $\hat{y} = U^T(x-\bar{x})$ | `proj = (X - mean_face) @ U` | Phép chiếu vuông góc xuống không gian eigenface. |
| 7 | $d = \|\hat{y} - \hat{y}_i\|_2$ | `np.sqrt(np.sum((p - q)**2))` | Khoảng cách Euclidean. |
| 8 | $\arg\min_i d_i$ | `np.argmin(distances)` | Chọn láng giềng gần nhất (1-NN). |
| 9 | $U^T U = I$ | `U.T @ U` (kết quả ≈ I) | Kiểm tra hệ cơ sở trực chuẩn. |

---

## 8. CHU TRÌNH CHẠY DỰ ÁN (gợi ý cho người mới)

```
# Bước A: Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# Bước B: Chạy toàn bộ pipeline (sinh ra biểu đồ + báo cáo)
python main_projection.py

# Bước C: Xem ví dụ tính tay từng bước (in chi tiết ra terminal)
python manual_example.py

# Bước D: Mở giao diện web demo (trên trình duyệt)
streamlit run web/streamlit_app.py
```

Kết quả sau khi chạy `main_projection.py`:
- Báo cáo so sánh được in ra terminal.
- 5 biểu đồ `.png` được lưu trong thư mục `outputs/`.

---

## 9. GHI CHÚ DÀNH CHO NGƯỜI MỚI

- **Vector hóa ảnh**: ảnh 2D `(112, 92)` được duỗi thành vector 1D `(10304,)` để đại số tuyến tính có thể xử lý.
- **Eigenfaces**: là các "hướng biến thiên" trong tập ảnh — eigenface đầu tiên giữ nhiều thông tin nhất.
- **Phép chiếu vuông góc**: nhân `U^T` với `(x - x̄)` để "ép" ảnh từ 10304 chiều xuống `k` chiều (vd: k=50).
- **1-NN**: sau khi chiếu, ta chỉ cần so khoảng cách trong không gian `k` chiều thay vì 10304 chiều → nhanh và chính xác hơn.
- **`np.linalg.eigh` vs `np.linalg.eig`**: `eigh` dành riêng cho ma trận đối xứng, ổn định số học hơn và đảm bảo trị riêng là số thực.
