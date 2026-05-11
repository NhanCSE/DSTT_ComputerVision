# YÊU CẦU HIỆN THỰC BÀI TẬP LỚN: NHẬN DẠNG ĐỐI TƯỢNG BẰNG PHÉP CHIẾU VUÔNG GÓC

## 1. MỤC TIÊU DỰ ÁN (PROJECT GOALS)
Bạn là một AI Assistant chuyên gia về Khoa học Dữ liệu và Đại số tuyến tính. [cite_start]Nhiệm vụ của bạn là hiện thực một chương trình nhận dạng đối tượng (cụ thể: Nhận dạng khuôn mặt - Eigenfaces) bằng ngôn ngữ Python[cite: 35, 51]. 
[cite_start]Chương trình này nhằm phục vụ cho Bài tập lớn môn Đại số tuyến tính, do đó cốt lõi của việc lập trình không phải là tạo ra một hệ thống sản phẩm (production-ready), mà là để minh họa trực quan các khái niệm toán học một cách minh bạch nhất[cite: 20].

## 2. RÀNG BUỘC PHẢI TUÂN THỦ NGHIÊM NGẶT (STRICT CONSTRAINTS)
- [cite_start]**Cấm sử dụng "Hộp đen" (No Black-box Functions):** Tuyệt đối KHÔNG sử dụng các hàm có sẵn để giải quyết trọng tâm bài toán (ví dụ: cấm dùng `sklearn.decomposition.PCA`)[cite: 33]. [cite_start]Mọi thao tác tính toán ma trận hiệp phương sai, trị riêng, vector riêng và phép chiếu vuông góc phải được code "chay" (from scratch) bằng `numpy`[cite: 33].
- **Tính giải thích (Explainability):** Thêm comment chi tiết vào mọi khối code toán học. [cite_start]Tác giả cần hiểu rõ từng dòng code để giải thích lại cho toàn bộ các thành viên trong nhóm[cite: 34, 43, 50].
- [cite_start]**Minh bạch nguồn gốc:** Bất kỳ thuật toán hay đoạn code nào tham khảo từ bên ngoài phải được chú thích nguồn rõ ràng dạng comment trong code[cite: 34, 50].

## 3. CƠ SỞ TOÁN HỌC (MATHEMATICAL FOUNDATION)
AI cần hiện thực chính xác các bước toán học sau:
1. **Tiền xử lý (Vector hóa):** Chuyển tập ảnh huấn luyện thành ma trận $A = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N]$. Tính ảnh trung bình $\mathbf{\bar{x}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$ và ma trận chuẩn hóa $\Phi_i = \mathbf{x}_i - \mathbf{\bar{x}}$.
2. **Không gian đặc trưng:** Tính ma trận hiệp phương sai $C$. Sử dụng SVD hoặc phân rã trị riêng (Eigendecomposition) để tìm tập vector riêng (eigenvectors) $U$.
3. **Phép chiếu vuông góc (Orthogonal Projection):** Chiếu một ảnh mới $\mathbf{y}$ xuống không gian con sinh bởi $U$: $\hat{\mathbf{y}} = U^T (\mathbf{y} - \mathbf{\bar{x}})$.
4. **Khoảng cách:** Nhận dạng bằng cách tính khoảng cách Euclidean giữa $\hat{\mathbf{y}}$ và các hình chiếu của tập ảnh huấn luyện.

## 4. CÁC BƯỚC HIỆN THỰC (IMPLEMENTATION STEPS)
Hãy tạo dự án với cấu trúc và thứ tự sau:

### Bước 1: Chuẩn bị Dataset & Dataloader
- Tải tập dữ liệu chuẩn (như ORL Faces hoặc Yale Face Database).
- Viết hàm đọc ảnh, chuyển sang grayscale và duỗi thành vector 1D.

### Bước 2: Hiện thực thuật toán Core (Tự viết)
- Khởi tạo class `OrthogonalFaceRecognizer`.
- Cài đặt phương thức `fit(X_train)`: Tính mean, covariance matrix, eigenvalues, eigenvectors. Sắp xếp và chọn $k$ vector riêng tốt nhất.
- Cài đặt phương thức `project(X)`: Thực hiện phép chiếu vuông góc.

### Bước 3: So sánh & Đánh giá (Bonus Points)
- Hiện thực thêm một phương pháp cơ bản (ví dụ: KNN so sánh từng pixel mà không dùng phép chiếu) để làm mốc đối chiếu (Baseline).
- [cite_start]Chạy đánh giá độ chính xác (Accuracy) trên tập Test giữa phương pháp "Phép chiếu vuông góc" và "Baseline" để có căn cứ viết báo cáo so sánh[cite: 31].

### Bước 4: Trực quan hóa (Visualization)
Dùng `matplotlib` sinh ra các biểu đồ sau để đưa vào slide và file báo cáo:
- Hình ảnh "Khuôn mặt trung bình" (Mean Face).
- Hình ảnh của một số "Khuôn mặt đặc trưng" (Eigenfaces / Vectors riêng).
- [cite_start]Minh họa một ví dụ nhận dạng đúng và một ví dụ nhận dạng sai[cite: 20].

### Bước 5: Ví dụ minh họa tính tay (Manual Example for Report)
Tạo file riêng `manual_example.py` hoặc section trong báo cáo minh họa thuật toán trên dữ liệu nhỏ (ví dụ: 3–4 ảnh 4×4 pixel):
- Tính mean vector từng bước.
- Tính ma trận hiệp phương sai bằng tay.
- Tìm 2 eigenvector đầu tiên.
- Chiếu một ảnh mới và tính khoảng cách nhận dạng.
- In ra từng ma trận trung gian để người đọc theo dõi được từng bước.

Mục đích: cung cấp ví dụ cụ thể (không phải abstract) cho phần **Thực hành** của báo cáo, tương ứng yêu cầu "Có ví dụ minh họa cho thuật toán" trong đề.

### Bước 6: Ứng dụng mở rộng (Extended Applications)
Hiện thực thêm 2 ứng dụng khác của phép chiếu vuông góc trong thị giác máy tính (vì đề tài được phân công bao gồm cả tái tạo và làm mờ hình ảnh):

**Ứng dụng 1 — Tái tạo ảnh (Image Reconstruction):**
- Chiếu ảnh xuống không gian con $k$ chiều rồi chiếu ngược lại: $\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \mathbf{\bar{x}}) + \mathbf{\bar{x}}$.
- Vẽ biểu đồ so sánh ảnh gốc vs ảnh tái tạo với $k = 5, 20, 50, 100$ eigenfaces.
- Tính PSNR hoặc MSE để đo chất lượng tái tạo theo $k$.

**Ứng dụng 2 — Làm mờ / Khử nhiễu ảnh (Image Denoising via Low-rank Projection):**
- Thêm nhiễu Gaussian vào ảnh gốc.
- Chiếu ảnh nhiễu xuống không gian con $k$ chiều (giữ lại các thành phần chính, loại bỏ nhiễu).
- So sánh trực quan ảnh gốc, ảnh nhiễu, và ảnh sau khi chiếu.

Hai ứng dụng này đáp ứng yêu cầu: *"Nếu có ứng dụng trong nhiều lĩnh vực khác nhau sẽ được đánh giá cao hơn"*.

## 5. LỆNH THỰC THI CHO AI (AI EXECUTION COMMAND)
Bắt đầu bằng việc tạo cấu trúc thư mục, tệp `requirements.txt` (chỉ chứa `numpy`, `matplotlib`, `Pillow`, `opencv-python`), và các file mã nguồn:
- `main_projection.py`: toàn bộ luồng nhận dạng khuôn mặt (Bước 1–4).
- `manual_example.py`: ví dụ tính tay trên dữ liệu nhỏ (Bước 5).
- `extended_applications.py`: tái tạo ảnh và làm mờ/khử nhiễu (Bước 6).