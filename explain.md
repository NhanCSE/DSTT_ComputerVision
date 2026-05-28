# GIẢI THÍCH CHI TIẾT TOÀN BỘ DỰ ÁN — TỪ A ĐẾN Z

> **Dành cho ai?** File này được viết cho người **chưa hề có kiến thức trước** về dự án, về Đại số tuyến tính nâng cao, hoặc về thị giác máy tính. Đọc xong file này, bạn sẽ hiểu:
> - Đề bài yêu cầu gì
> - Tại sao lại làm như vậy (lý thuyết toán học)
> - Cấu trúc dự án ra sao
> - Mỗi dòng code làm gì
> - Kết quả ra sao và phải giải thích thế nào khi bị hỏi

---

## MỤC LỤC

1. [Đề bài và yêu cầu](#1-đề-bài-và-yêu-cầu)
2. [Tổng quan dự án](#2-tổng-quan-dự-án)
3. [Kiến thức nền tảng cần biết](#3-kiến-thức-nền-tảng-cần-biết)
4. [Cơ sở toán học — Phép chiếu vuông góc](#4-cơ-sở-toán-học--phép-chiếu-vuông-góc)
5. [Thuật toán Eigenfaces — Trái tim của dự án](#5-thuật-toán-eigenfaces--trái-tim-của-dự-án)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Cài đặt và chạy dự án](#7-cài-đặt-và-chạy-dự-án)
8. [Giải thích từng file code](#8-giải-thích-từng-file-code)
9. [Ba ứng dụng — Hiểu sâu](#9-ba-ứng-dụng--hiểu-sâu)
10. [Kết quả kỳ vọng và cách đọc biểu đồ](#10-kết-quả-kỳ-vọng-và-cách-đọc-biểu-đồ)
11. [Câu hỏi thường gặp khi báo cáo](#11-câu-hỏi-thường-gặp-khi-báo-cáo)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. ĐỀ BÀI VÀ YÊU CẦU

### 1.1 Đề bài gốc

Đây là **Bài tập lớn môn Đại số Tuyến tính**. Đề tài được phân công:

> **"Nghiên cứu tính chất và ứng dụng của phép chiếu vuông góc trong thị giác máy tính."**
>
> **Ứng dụng cụ thể:** Nhận dạng đối tượng, tái tạo hình ảnh, và kỹ thuật làm mờ hình ảnh.

### 1.2 Tại sao đề tài này?

Đại số tuyến tính có một khái niệm rất quan trọng tên là **"phép chiếu vuông góc"** (orthogonal projection). Nó nghe trừu tượng nhưng thực ra cực kỳ hữu ích — nó là nền tảng của hầu hết các thuật toán **nén dữ liệu, học máy, xử lý ảnh, xử lý tín hiệu** hiện đại.

Đề bài muốn chúng ta:
- **Hiểu** phép chiếu vuông góc là gì về mặt toán học
- **Chứng minh** được tại sao nó có những tính chất hay (như "giữ lại được nhiều thông tin nhất có thể")
- **Áp dụng** nó vào 3 bài toán cụ thể trong xử lý ảnh để thấy rõ sức mạnh

### 1.3 Quy định nghiêm ngặt từ giáo viên

Đề bài yêu cầu:

| Yêu cầu | Giải thích |
|---|---|
| **KHÔNG dùng "hộp đen"** | Không được gọi thẳng `sklearn.decomposition.PCA` hay các hàm ML có sẵn. Phải tự code các phép tính ma trận (hiệp phương sai, trị riêng, vector riêng, phép chiếu) bằng `numpy` thuần. |
| **Phải hiểu từng dòng code** | Giáo viên có thể hỏi ngẫu nhiên 1 thành viên. Nếu không trả lời được thì cả nhóm bị trừ điểm. |
| **Trích dẫn nguồn rõ ràng** | Code tham khảo từ đâu phải ghi nguồn dưới dạng comment. |
| **Ít nhất một bài toán thực tế** | Phải áp dụng vào dữ liệu thật (không phải dữ liệu giả lập). |
| **Đánh giá cao nếu so sánh nhiều phương pháp** | Vì vậy dự án có thêm phần Baseline để so sánh. |

### 1.4 Điểm số được chia thế nào?

| Tiêu chí | % điểm |
|---|---|
| Nộp bài + điểm danh | 10% |
| Bài báo cáo (lý thuyết + thuật toán + code + trình bày) | 50% |
| Hỏi đáp khi báo cáo | 40% |

Nghĩa là **chỉ riêng phần hỏi đáp đã chiếm 40%** — phải hiểu thật sâu, không học vẹt.

---

## 2. TỔNG QUAN DỰ ÁN

### 2.1 Dự án làm cái gì?

Dự án có 1 thuật toán **lõi** + 3 **ứng dụng** dùng cùng thuật toán đó.

**Thuật toán lõi:** Thuật toán **Eigenfaces** (Turk & Pentland, 1991) — chính là một ứng dụng kinh điển của phép chiếu vuông góc.

**3 ứng dụng:**

| # | Ứng dụng | Bài toán |
|---|---|---|
| 1 | **Nhận dạng khuôn mặt** | Cho 1 ảnh mặt mới, máy tính trả lời "đây là ai?" |
| 2 | **Tái tạo ảnh** | Nén ảnh xuống cực nhỏ rồi giải nén — xem chất lượng sau khi nén |
| 3 | **Làm mờ ảnh** | Làm mờ ảnh bằng cách bỏ chi tiết tần số cao — như một bộ lọc nghệ thuật |

**Điều thú vị nhất:** Cả 3 ứng dụng đều dùng **CÙNG MỘT công thức toán học** — chỉ thay đổi cách dùng kết quả. Đây là vẻ đẹp của Đại số tuyến tính.

### 2.2 Dataset (Tập dữ liệu)

Dự án dùng **AT&T Database of Faces** (còn gọi là **ORL Faces Database**):
- **40 người** khác nhau
- **10 ảnh/người** (chụp ở các góc, biểu cảm khác nhau)
- **Tổng cộng: 400 ảnh**
- Mỗi ảnh: **92 × 112 pixel**, grayscale (ảnh đen trắng)
- Nguồn: AT&T Laboratories Cambridge (Anh)
- File ảnh dạng `.pgm` (định dạng ảnh đen trắng đơn giản)

**Cách chia dữ liệu (rất quan trọng):**
- **Tập huấn luyện (training):** 8 ảnh đầu/người → **320 ảnh** → dùng để "dạy" máy
- **Tập kiểm tra (test):** 2 ảnh cuối/người → **80 ảnh** → dùng để "kiểm tra" xem máy học được gì

Lý do chia: máy không được "thấy" ảnh test trong lúc học. Khi kiểm tra, ta dùng ảnh test để xem máy có nhận dạng đúng người đó không.

### 2.3 Pipeline tổng thể (6 bước)

```
[Bước 1] Tải dataset → vector hóa → chia train/test
   ↓
[Bước 2] Huấn luyện thuật toán Eigenfaces
   ↓
[Bước 3] So sánh Eigenfaces với phương pháp baseline đơn giản (Pixel-KNN)
   ↓
[Bước 4] Vẽ 5 biểu đồ minh họa cho báo cáo
   ↓
[Bước 5] Ví dụ tính tay trên dữ liệu nhỏ (4 ảnh 3×3) để báo cáo hiểu rõ
   ↓
[Bước 6] Hai ứng dụng mở rộng: Tái tạo ảnh + Làm mờ ảnh
```

---

## 3. KIẾN THỨC NỀN TẢNG CẦN BIẾT

Trước khi đi vào lý thuyết chính, bạn cần biết các khái niệm cơ bản sau. Nếu đã biết rồi, có thể bỏ qua mục này.

### 3.1 Ảnh là gì trong máy tính?

Một bức ảnh grayscale (đen trắng) đối với máy tính chỉ là một **ma trận số**. Mỗi ô của ma trận là một **pixel**, giá trị từ **0** (đen tuyền) đến **255** (trắng tinh).

Ví dụ: ảnh 3×3 pixel:
```
[ 10,  20,  30]
[ 40, 100, 200]
[150, 200, 250]
```

Ảnh ORL có kích thước 112 hàng × 92 cột = **10304 pixel** mỗi ảnh.

### 3.2 Vector hóa ảnh

Để máy tính dễ tính toán, ta **"duỗi thẳng"** ảnh 2D thành một dãy số dài (vector 1D). Đọc theo từng hàng, ghép lại thành 1 dòng:

```
Ảnh 2D:                Vector 1D (duỗi theo hàng):
[ 10,  20,  30]
[ 40, 100, 200]   →    [10, 20, 30, 40, 100, 200, 150, 200, 250]
[150, 200, 250]
```

Với ảnh ORL: ảnh 112×92 → vector dài **10304 số**.

→ Mỗi ảnh trở thành **một điểm trong không gian 10304 chiều** $\mathbb{R}^{10304}$.

### 3.3 Vector, ma trận, không gian — Tóm tắt cực ngắn

- **Vector** = một dãy số. Ví dụ: $(3, 5, 1)$ là vector 3 chiều.
- **Ma trận** = một bảng số, có hàng và cột. Vector chính là ma trận có 1 hàng (hoặc 1 cột).
- **Không gian $\mathbb{R}^n$** = tập hợp tất cả các vector $n$ chiều.
- **Tích vô hướng** (dot product) của 2 vector $\mathbf{a}, \mathbf{b}$:
  $$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \ldots + a_n b_n$$
  Nó cho ta biết 2 vector "giống nhau" cỡ nào.
- **Độ dài (norm)** của vector $\mathbf{v} = (v_1, \ldots, v_n)$:
  $$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2}$$
  Định lý Pythagoras tổng quát.
- **Khoảng cách Euclidean** giữa 2 vector: chính là độ dài của hiệu hai vector.
  $$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|$$
- **Vector trực giao**: hai vector vuông góc với nhau ⟺ tích vô hướng = 0.
- **Vector đơn vị (unit vector)**: vector có độ dài = 1.
- **Hệ trực chuẩn (orthonormal)**: tập các vector đôi một trực giao và mỗi vector đều có độ dài 1. Ví dụ: $\{(1,0,0), (0,1,0), (0,0,1)\}$ trong $\mathbb{R}^3$.

### 3.4 Trị riêng và vector riêng (eigenvalue & eigenvector)

Đây là khái niệm **cốt lõi** của dự án. Cần hiểu thật kỹ.

**Định nghĩa:** Cho ma trận vuông $A$. Vector $\mathbf{v} \neq \mathbf{0}$ được gọi là **vector riêng** của $A$ nếu:
$$A \mathbf{v} = \lambda \mathbf{v}$$
trong đó $\lambda$ là một số (gọi là **trị riêng** ứng với $\mathbf{v}$).

**Ý nghĩa trực quan:** Bình thường, khi ta nhân ma trận $A$ với vector $\mathbf{v}$, kết quả $A\mathbf{v}$ là một vector khác — có thể khác cả về hướng lẫn độ dài. Nhưng nếu $\mathbf{v}$ là vector riêng, thì $A\mathbf{v}$ **vẫn cùng hướng** với $\mathbf{v}$, chỉ co giãn lên/xuống $\lambda$ lần.

**Hình ảnh hóa:** Tưởng tượng $A$ là một phép biến đổi không gian (kéo, nén, xoay…). Vector riêng là những hướng "đặc biệt" — hướng mà phép biến đổi chỉ làm chúng dài/ngắn ra chứ không làm xoay.

**Tại sao quan trọng?** Trong dữ liệu ảnh, **các vector riêng của ma trận hiệp phương sai** chính là **các hướng mà dữ liệu biến thiên nhiều nhất**. Đó chính là **eigenfaces** trong dự án này.

### 3.5 Hiệp phương sai (Covariance)

Hiệp phương sai cho biết **2 biến biến thiên cùng nhau ra sao**.

Với một tập $N$ ảnh đã được vector hóa $\mathbf{x}_1, \ldots, \mathbf{x}_N$ (mỗi ảnh là vector $p$ chiều):

**Bước 1: Tính trung bình:**
$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i$$

**Bước 2: Trung tâm hóa (centering):**
$$\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}$$
Mỗi $\Phi_i$ là "phần lệch" của ảnh thứ $i$ so với khuôn mặt trung bình.

**Bước 3: Ma trận hiệp phương sai:**
$$C = \frac{1}{N} \Phi^T \Phi$$
(với $\Phi$ là ma trận có các $\Phi_i$ là hàng).

$C$ là ma trận $p \times p$. Phần tử $C_{ij}$ cho biết pixel thứ $i$ và pixel thứ $j$ biến thiên cùng nhau ra sao trên toàn bộ tập ảnh.

### 3.6 Tóm gọn: cần nhớ những gì?

- Ảnh → vector → điểm trong không gian nhiều chiều
- Trị riêng/vector riêng cho biết "hướng quan trọng nhất" của dữ liệu
- Ma trận hiệp phương sai mô tả cách dữ liệu biến thiên

Bây giờ ta vào phần chính.

---

## 4. CƠ SỞ TOÁN HỌC — PHÉP CHIẾU VUÔNG GÓC

### 4.1 Phép chiếu vuông góc là gì? (Trực quan)

**Trong không gian 2D:** Tưởng tượng một điểm $P$ và một đường thẳng $\ell$. **Phép chiếu vuông góc** của $P$ lên $\ell$ là điểm trên $\ell$ gần $P$ nhất — chính là điểm bạn vẽ được khi hạ đường vuông góc từ $P$ xuống $\ell$.

```
       P  •
         /|
        / |
       /  | (đường vuông góc)
      /   |
─────●────●─────────────  ℓ
    chiếu(P)
```

**Trong không gian 3D:** Chiếu một điểm lên một mặt phẳng. Vẫn là điểm gần nhất trên mặt phẳng.

**Tổng quát ($\mathbb{R}^p$):** Chiếu một vector $\mathbf{y} \in \mathbb{R}^p$ lên một không gian con $S$ (chiều thấp hơn) — kết quả là vector trong $S$ gần $\mathbf{y}$ nhất.

### 4.2 Công thức phép chiếu vuông góc

Giả sử ta có $k$ vector trực chuẩn $\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k$ (đôi một vuông góc, mỗi cái có độ dài 1). Chúng tạo thành cơ sở của một không gian con $S \subset \mathbb{R}^p$.

Gộp chúng vào ma trận:
$$U_k = [\mathbf{u}_1 \mid \mathbf{u}_2 \mid \cdots \mid \mathbf{u}_k] \in \mathbb{R}^{p \times k}$$

**Phép chiếu vuông góc của $\mathbf{y}$ lên $S$:**
$$\text{proj}_S(\mathbf{y}) = U_k U_k^T \mathbf{y}$$

**Tọa độ của $\mathbf{y}$ trong cơ sở mới:**
$$\hat{\mathbf{y}} = U_k^T \mathbf{y} \in \mathbb{R}^k$$

(Chú ý: $\hat{\mathbf{y}}$ chỉ có $k$ số — ít hơn $p$ rất nhiều. Đây là điều quan trọng!)

### 4.3 Tại sao $U_k^T U_k = I$ và sao điều đó tốt?

Vì các cột của $U_k$ trực chuẩn nên:
$$U_k^T U_k = I_k \quad \text{(ma trận đơn vị } k \times k\text{)}$$

**Hệ quả:**
- Chiếu rồi chiếu ngược: $U_k U_k^T (U_k \hat{\mathbf{y}}) = U_k \hat{\mathbf{y}}$ — không méo mó.
- Khoảng cách trong không gian con bằng khoảng cách trong không gian gốc (sau khi đã chiếu): rất quan trọng cho việc nhận dạng.

### 4.4 Tính chất quan trọng nhất: phép chiếu là "xấp xỉ tối ưu"

**Định lý (Best Approximation Theorem):** Trong tất cả các vector $\mathbf{w} \in S$, vector $\text{proj}_S(\mathbf{y})$ là vector **gần $\mathbf{y}$ nhất**:
$$\|\mathbf{y} - \text{proj}_S(\mathbf{y})\| \leq \|\mathbf{y} - \mathbf{w}\| \quad \forall \mathbf{w} \in S$$

**Ý nghĩa với dự án:** Khi ta chiếu một ảnh xuống không gian $k$ chiều, ta đang chọn **biểu diễn gần đúng nhất có thể** của ảnh đó bằng $k$ con số. Đó chính là cách dữ liệu được "nén" thông minh nhất.

### 4.5 Sai số tái tạo

$$\text{sai số} = \mathbf{y} - \text{proj}_S(\mathbf{y})$$

Sai số này luôn **vuông góc** với không gian $S$ (đây là lý do gọi là "phép chiếu vuông góc"). Phần sai số chính là "thông tin bị bỏ đi" khi nén.

---

## 5. THUẬT TOÁN EIGENFACES — TRÁI TIM CỦA DỰ ÁN

### 5.1 Ý tưởng tổng quan

Mọi khuôn mặt người đều có những điểm chung (hai mắt, một mũi, một miệng) — đó là "khuôn mặt trung bình". Nhưng giữa các khuôn mặt cũng có những khác biệt — và những khác biệt đó **có CẤU TRÚC**: chúng tập trung vào một vài "hướng biến thiên" chính trong không gian ảnh.

**Eigenfaces** là tên gọi cho các **vector riêng của ma trận hiệp phương sai** của tập ảnh huấn luyện. Mỗi eigenface bản thân nó cũng là một ảnh (vì cùng số chiều với ảnh đầu vào). Nó biểu diễn **một hướng biến thiên** của bộ dữ liệu.

**Phép thuật:** Thay vì lưu ảnh bằng 10304 pixel, ta chỉ cần lưu **tọa độ của ảnh trong không gian eigenface** — chỉ cần $k$ con số (thường $k = 50$). Đó là cách Eigenfaces vừa nén dữ liệu vừa giữ được thông tin nhận dạng.

### 5.2 Pipeline thuật toán (chi tiết từng bước)

#### Bước A: Vector hóa ảnh

Mỗi ảnh $\mathbf{x}_i$ là vector $p = 10304$ chiều. Tập $N = 320$ ảnh train:
$$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix} \in \mathbb{R}^{N \times p}$$

#### Bước B: Tính khuôn mặt trung bình

$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i$$

→ Trung bình theo từng pixel. Khi reshape lại 2D, $\bar{\mathbf{x}}$ trông như một khuôn mặt mờ — đó là "khuôn mặt trung bình của 40 người".

#### Bước C: Trung tâm hóa

$$\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}$$

Gộp lại thành ma trận $\Phi \in \mathbb{R}^{N \times p}$. Bây giờ trung bình của các $\Phi_i$ bằng 0 — dữ liệu đã được "căn giữa".

#### Bước D: Ma trận hiệp phương sai — và một thủ thuật quan trọng!

**Theo lý thuyết:**
$$C = \frac{1}{N}\Phi^T \Phi \in \mathbb{R}^{p \times p}$$

**Vấn đề:** $C$ có kích thước $10304 \times 10304$ — quá to (≈ 800 MB RAM) và việc tìm trị riêng cực kỳ chậm (mất hàng giờ).

**Thủ thuật Turk & Pentland (1991):** Tính ma trận "thay thế" nhỏ hơn rất nhiều:
$$L = \frac{1}{N} \Phi \Phi^T \in \mathbb{R}^{N \times N}$$

Với $N = 320$, $L$ chỉ là ma trận $320 \times 320$ — tìm trị riêng trong vài giây.

**Chứng minh thủ thuật đúng:**

Giả sử $\mathbf{v}$ là vector riêng của $L$ với trị riêng $\lambda$:
$$L \mathbf{v} = \lambda \mathbf{v}$$
$$\frac{1}{N} \Phi \Phi^T \mathbf{v} = \lambda \mathbf{v}$$

Nhân hai vế bên trái với $\Phi^T$:
$$\frac{1}{N} \Phi^T \Phi \Phi^T \mathbf{v} = \lambda \Phi^T \mathbf{v}$$
$$C \cdot \underbrace{(\Phi^T \mathbf{v})}_{\mathbf{u}} = \lambda \cdot (\Phi^T \mathbf{v})$$

Vậy $\mathbf{u} = \Phi^T \mathbf{v}$ là **vector riêng của $C$** với **cùng trị riêng $\lambda$**! ✓

→ Thay vì phân rã $C$ ($10304 \times 10304$), ta phân rã $L$ ($320 \times 320$) rồi khôi phục lại $\mathbf{u}_i = \Phi^T \mathbf{v}_i$.

#### Bước E: Phân rã trị riêng của $L$

Dùng `numpy.linalg.eigh()` (vì $L$ đối xứng):
$$L = V \Lambda V^T$$

trong đó:
- $V$: ma trận có các cột là vector riêng của $L$
- $\Lambda$: ma trận đường chéo chứa các trị riêng $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N$

**Lý do dùng `eigh()` chứ không phải `eig()`:**
- `eigh()` dành riêng cho ma trận đối xứng → ổn định số học hơn
- Đảm bảo trị riêng là số thực (không có phần ảo do sai số tính toán)
- Tham khảo: [NumPy docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)

#### Bước F: Khôi phục eigenfaces

Mỗi $\mathbf{u}_i = \Phi^T \mathbf{v}_i$ là vector riêng của $C$, nhưng chưa có độ dài 1. Chuẩn hóa:
$$\mathbf{u}_i \leftarrow \frac{\mathbf{u}_i}{\|\mathbf{u}_i\|}$$

Gộp $k$ eigenfaces tốt nhất (eigenvalue lớn nhất):
$$U_k = [\mathbf{u}_1 \mid \mathbf{u}_2 \mid \cdots \mid \mathbf{u}_k] \in \mathbb{R}^{p \times k}$$

**Điều đặc biệt:** Mỗi cột $\mathbf{u}_i$ khi reshape thành 2D **trông như một khuôn mặt** — đó là lý do gọi là "eigenface" (khuôn mặt đặc trưng).

#### Bước G: Chiếu tập train xuống không gian eigenface

$$\hat{\mathbf{x}}_i = U_k^T (\mathbf{x}_i - \bar{\mathbf{x}}) \in \mathbb{R}^k$$

Mỗi ảnh train giờ chỉ còn $k$ con số. Lưu lại để dùng khi nhận dạng.

#### Bước H: Nhận dạng ảnh mới

Cho ảnh test $\mathbf{y}$:

1. Chiếu xuống không gian eigenface:
   $$\hat{\mathbf{y}} = U_k^T (\mathbf{y} - \bar{\mathbf{x}})$$

2. Tìm ảnh train gần nhất bằng khoảng cách Euclidean (1-Nearest Neighbor):
   $$i^* = \arg\min_i \|\hat{\mathbf{y}} - \hat{\mathbf{x}}_i\|$$

3. Nhãn của ảnh test = nhãn của ảnh train $i^*$.

### 5.3 Tại sao Eigenfaces hiệu quả?

1. **Giảm chiều mạnh:** Từ 10304 chiều → khoảng 50 chiều. Nhanh hơn nhiều khi so sánh.
2. **Loại bỏ nhiễu:** Các eigenface đầu tiên chỉ giữ "đặc trưng quan trọng nhất", loại bỏ các chi tiết nhỏ thường là nhiễu.
3. **Tương đồng có ý nghĩa:** Hai ảnh của cùng một người thường gần nhau trong không gian eigenface, dù ảnh gốc có khác biệt về ánh sáng, biểu cảm.
4. **Tính chất phép chiếu vuông góc:** Đây là cách "nén" thông tin tối ưu nhất theo nghĩa bình phương sai số nhỏ nhất.

### 5.4 Phép chiếu vuông góc xuất hiện ở đâu trong dự án?

| Vị trí | Vai trò |
|---|---|
| Tính eigenfaces | Eigenfaces chính là một cơ sở trực chuẩn |
| `project()` | $\hat{\mathbf{y}} = U_k^T(\mathbf{y} - \bar{\mathbf{x}})$ — phép chiếu cốt lõi |
| `reconstruct()` | $\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$ — chiếu xuống rồi chiếu ngược |
| Nhận dạng | So sánh khoảng cách trong không gian con (đã chiếu) |
| Tái tạo & làm mờ ảnh | Dùng phép chiếu để bỏ thông tin "thừa" |

→ **Toàn bộ dự án xoay quanh phép chiếu vuông góc** — đáp ứng đúng đề tài.

---

## 6. CẤU TRÚC THƯ MỤC

```
DSTT_ComputerVision/
│
├── README.md                  # Tổng quan dự án (ngắn gọn)
├── REQUIREMENT.md             # Đề bài gốc của giáo viên
├── IMPLEMENTATION_PLAN.md     # Kế hoạch hiện thực
├── YEU_CAU_THAY_DOI.md        # Ghi chú thay đổi: từ "khử nhiễu" → "làm mờ"
├── explain.md                 # ← FILE NÀY: giải thích chi tiết toàn bộ
├── requirements.txt           # Danh sách thư viện Python cần cài
│
├── main_projection.py         # Entry point — chạy toàn bộ pipeline
├── manual_example.py          # Bước 5 — ví dụ tính tay trên dữ liệu nhỏ
├── extended_applications.py   # Bước 6 — tái tạo ảnh + làm mờ ảnh
│
├── src/                       # Code lõi tách module hóa
│   ├── __init__.py
│   ├── dataloader.py          # Bước 1 — đọc dataset, vector hóa, chia train/test
│   ├── recognizer.py          # Bước 2 — class OrthogonalFaceRecognizer (TRỌNG TÂM)
│   ├── baseline.py            # Bước 3a — class PixelKNNRecognizer (so sánh)
│   ├── evaluator.py           # Bước 3b — đo accuracy, so sánh phương pháp
│   └── visualizer.py          # Bước 4 — 5 biểu đồ cho báo cáo
│
├── data/                      # Dataset (tự động tải lần đầu chạy)
│   └── orl_faces/
│       ├── s1/  (1.pgm … 10.pgm)   # 10 ảnh của người 1
│       ├── s2/  (1.pgm … 10.pgm)
│       └── …
│       └── s40/
│
└── outputs/                   # Tất cả biểu đồ output được lưu ở đây
    ├── mean_face.png
    ├── eigenfaces.png
    ├── recognition.png
    ├── accuracy_vs_k.png
    ├── variance_ratio.png
    ├── app1_reconstruction.png
    ├── app1_quality_curve.png
    ├── app2_blurring.png
    └── manual_example.png
```

**Triết lý tổ chức code:**
- Mỗi file/module có **một nhiệm vụ rõ ràng** (Single Responsibility Principle)
- Các module trong `src/` có thể chạy độc lập (đoạn `if __name__ == "__main__":` ở cuối mỗi file)
- File `main_projection.py` chỉ điều phối, không chứa thuật toán

---

## 7. CÀI ĐẶT VÀ CHẠY DỰ ÁN

### 7.1 Yêu cầu

- **Python ≥ 3.8** (project dùng cú pháp `int | None` của Python 3.10+ với `from __future__ import annotations` để tương thích ngược)
- **Kết nối Internet** (chỉ lần đầu để tải dataset ~1.7 MB)

### 7.2 Cài thư viện

```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:
```
numpy           # tính toán ma trận
matplotlib      # vẽ biểu đồ
Pillow          # đọc/xử lý ảnh
opencv-python   # (dự phòng — không thực sự dùng trong code hiện tại)
```

**KHÔNG có `scikit-learn`** — đúng theo yêu cầu của đề bài (không "hộp đen").

### 7.3 Các lệnh chạy

**Cách 1 — Chạy toàn bộ pipeline (KHUYẾN NGHỊ):**
```bash
python main_projection.py
```
→ Chạy Bước 1, 3a, 2, 3b, 4, 6 — sinh toàn bộ 8 biểu đồ vào `outputs/`.

**Cách 2 — Chạy ví dụ tính tay (Bước 5):**
```bash
python manual_example.py
```
→ In từng bước thuật toán ra console + lưu `outputs/manual_example.png`.

**Cách 3 — Chạy từng module độc lập (để debug/kiểm tra):**
```bash
python src/dataloader.py         # kiểm tra đọc dataset
python src/recognizer.py         # huấn luyện + đánh giá nhanh
python src/evaluator.py          # so sánh đầy đủ
python src/visualizer.py         # sinh toàn bộ biểu đồ
python extended_applications.py  # tái tạo + làm mờ ảnh
```

### 7.4 Thứ tự thực thi trong `main_projection.py`

```
[Bước 1]  Tải ORL Faces → vector hóa → chia train/test
[Bước 3a] Thử k = 5, 10, 20, 30, 50, 75, 100, 150 → chọn k tối ưu
[Bước 2]  Huấn luyện OrthogonalFaceRecognizer với k tối ưu
[Bước 3b] So sánh Eigenfaces vs Baseline Pixel-KNN → in báo cáo
[Bước 4]  Sinh 5 biểu đồ → outputs/
[Bước 6]  Huấn luyện lại với k=150 → tái tạo & làm mờ ảnh → thêm 3 biểu đồ
```

**Lưu ý vì sao chạy Bước 3a trước Bước 2:** Để chọn được $k$ tốt nhất cho mô hình chính thức, ta phải thử nhiều $k$ trước, rồi mới huấn luyện mô hình cuối cùng với $k$ tối ưu.

---

## 8. GIẢI THÍCH TỪNG FILE CODE

### 8.1 `src/dataloader.py` — Bước 1: Tải dữ liệu

**Mục đích:** Lấy 400 ảnh, biến mỗi ảnh thành vector 10304 chiều, chia thành 320 train + 80 test.

**Các hàm chính:**

#### `download_orl_dataset(save_dir)`
Tự động tải file `att_faces.zip` từ Internet, giải nén vào `data/orl_faces/`.
- Thử URL chính trước, nếu lỗi thử URL dự phòng (GitHub mirror).
- Nếu cả 2 đều thất bại, in hướng dẫn tải thủ công.
- Nếu dataset đã tồn tại → bỏ qua, không tải lại.

#### `load_image_as_vector(image_path)`
Đọc 1 file ảnh, chuyển thành vector 10304 chiều.

```python
img = Image.open(image_path)           # mở ảnh bằng Pillow
img_gray = img.convert("L")            # chuyển grayscale (mode "L" = 8-bit luminance)
img_resized = img_gray.resize((92, 112))   # resize chuẩn
img_array = np.array(img_resized, dtype=np.float64)  # → mảng 2D (112, 92)
img_vector = img_array.flatten()       # duỗi 1D → (10304,)
return img_vector
```

**Tại sao `float64`?** Để tính toán ma trận chính xác. Nếu để `uint8` (0-255 nguyên) thì các phép trừ, nhân có thể bị tràn số hoặc làm tròn sai.

#### `load_orl_dataset(...)`
Duyệt qua 40 thư mục `s1/, s2/, …, s40/`, mỗi thư mục có 10 ảnh.
- Ảnh 1-8 mỗi người → train (320 ảnh)
- Ảnh 9-10 mỗi người → test (80 ảnh)
- Gán nhãn `y` = số người (1-40)
- Trả về: `(X_train, y_train, X_test, y_test)` dưới dạng mảng numpy.

**Quy ước về shape:**
- `X_train.shape == (320, 10304)` — mỗi **HÀNG** là một ảnh
- `y_train.shape == (320,)` — vector nhãn

#### `print_dataset_info(...)`
In thống kê tóm tắt: kích thước ảnh, số người, shape các mảng, miền giá trị pixel.

---

### 8.2 `src/recognizer.py` — Bước 2: Thuật toán Eigenfaces (TRỌNG TÂM)

**File quan trọng nhất** — chứa class `OrthogonalFaceRecognizer` hiện thực phép chiếu vuông góc từ A đến Z.

#### Cấu trúc class

```python
class OrthogonalFaceRecognizer:
    def __init__(self, n_components=None):
        # n_components = k = số eigenfaces giữ lại
        # Nếu None, giữ tất cả

        self.mean_face_         = None   # x̄, shape (p,)
        self.eigenfaces_        = None   # U, shape (p, k)
        self.eigenvalues_       = None   # λ, shape (k,)
        self.train_projections_ = None   # ŷ của tập train
        self.train_labels_      = None
```

#### Phương thức `fit(X_train, y_train)` — Huấn luyện

Đây là nơi hiện thực **7 bước thuật toán** ở mục 5.2:

```python
def fit(self, X_train, y_train):
    N, p = X_train.shape

    # BƯỚC 2.1: Khuôn mặt trung bình
    self.mean_face_ = np.mean(X_train, axis=0)
    # axis=0 → trung bình theo CỘT (theo từng pixel, qua tất cả ảnh)

    # BƯỚC 2.2: Trung tâm hóa
    Phi = X_train - self.mean_face_
    # Broadcasting: numpy tự trừ mean_face_ khỏi MỖI hàng của X_train

    # BƯỚC 2.3: Thủ thuật Turk & Pentland — tính L thay vì C
    L = (Phi @ Phi.T) / N
    # L = (320, 320), thay vì C = (10304, 10304)

    # BƯỚC 2.4: Phân rã trị riêng của L
    eigenvalues_L, V = np.linalg.eigh(L)
    # eigenvalues_L: trị riêng (tăng dần)
    # V: ma trận có các cột là vector riêng

    # Đảo ngược để có thứ tự giảm dần
    idx = np.argsort(eigenvalues_L)[::-1]
    eigenvalues_L = eigenvalues_L[idx]
    V = V[:, idx]

    # Loại bỏ trị riêng âm/0 (do sai số số học)
    valid = eigenvalues_L > 1e-10
    eigenvalues_L = eigenvalues_L[valid]
    V = V[:, valid]

    # BƯỚC 2.5: Khôi phục eigenfaces của C
    U_unnorm = Phi.T @ V   # u_i = Φ^T · v_i
    norms = np.linalg.norm(U_unnorm, axis=0, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # tránh chia 0
    U = U_unnorm / norms

    # BƯỚC 2.6: Chọn k eigenfaces tốt nhất
    k = self.n_components if self.n_components is not None else U.shape[1]
    k = min(k, U.shape[1])
    self.eigenvalues_ = eigenvalues_L[:k]
    self.eigenfaces_  = U[:, :k]

    # BƯỚC 2.7: Chiếu tập train (lưu lại để predict)
    self.train_projections_ = self.project(X_train)
    self.train_labels_      = y_train.copy()

    return self
```

**Trick numpy quan trọng cần hiểu:**

| Cú pháp | Ý nghĩa |
|---|---|
| `A @ B` | Nhân ma trận (giống `np.matmul`) |
| `A.T` | Chuyển vị (transpose) |
| `X - v` (X 2D, v 1D) | **Broadcasting**: trừ v khỏi mỗi hàng của X |
| `np.argsort(a)[::-1]` | Sắp xếp giảm dần |
| `np.linalg.eigh(M)` | Trị riêng + vector riêng (cho ma trận đối xứng) |
| `np.linalg.norm(M, axis=0)` | Độ dài Euclidean theo từng CỘT |
| `M[:, idx]` | Lấy các cột theo chỉ số `idx` |

#### Phương thức `project(X)` — Phép chiếu vuông góc

```python
def project(self, X):
    """Công thức: ŷ = U^T (x - x̄)"""
    single = X.ndim == 1
    if single:
        X = X.reshape(1, -1)   # cho phép truyền vào 1 vector hoặc nhiều vector

    X_centered = X - self.mean_face_                 # (N, p)
    projections = X_centered @ self.eigenfaces_      # (N, p) @ (p, k) = (N, k)
    # Mỗi hàng của projections là ŷ của một ảnh

    return projections[0] if single else projections
```

**Tại sao `X_centered @ eigenfaces_` chứ không phải `eigenfaces_.T @ X_centered`?**

Vì `X_centered` được lưu với mỗi HÀNG là một ảnh, không phải mỗi CỘT. Nên:
- Theo công thức gốc: $\hat{\mathbf{y}} = U^T \mathbf{x}_\text{centered}$ (vector cột)
- Khi $\mathbf{x}_\text{centered}$ là hàng: $\hat{\mathbf{y}}^T = \mathbf{x}_\text{centered}^T (U^T)^T = \mathbf{x}_\text{centered}^T U$

→ Trong numpy với layout "row-major", viết `X_centered @ U` chính là làm điều này.

#### Phương thức `reconstruct(X, n_components)` — Tái tạo ảnh

```python
def reconstruct(self, X, n_components=None):
    """Công thức: x̂ = U_k · U_k^T · (x - x̄) + x̄"""
    k = n_components if n_components is not None else self.eigenfaces_.shape[1]
    k = min(k, self.eigenfaces_.shape[1])
    U_k = self.eigenfaces_[:, :k]

    X_centered = X - self.mean_face_
    coords = X_centered @ U_k              # chiếu xuống (= ŷ)
    X_reconstructed = coords @ U_k.T + self.mean_face_   # chiếu ngược + cộng x̄

    X_reconstructed = np.clip(X_reconstructed, 0, 255)
    return X_reconstructed
```

**Tại sao phải `np.clip(_, 0, 255)`?** Vì sau khi tái tạo, giá trị pixel có thể hơi nhỏ hơn 0 hoặc lớn hơn 255 do sai số. Cần "kẹp" lại trong vùng hợp lệ.

#### Phương thức `predict(X_test)` — Nhận dạng (1-NN)

```python
def predict(self, X_test):
    test_projs = self.project(X_test)        # chiếu tập test → (N_test, k)

    N_test = test_projs.shape[0]
    predictions = np.zeros(N_test, dtype=np.int32)

    for i in range(N_test):
        # Khoảng cách Euclidean tới mỗi ảnh train
        diffs     = self.train_projections_ - test_projs[i]   # (N_train, k)
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))       # (N_train,)

        # Ảnh train gần nhất
        nearest_idx    = np.argmin(distances)
        predictions[i] = self.train_labels_[nearest_idx]

    return predictions
```

#### Các phương thức tiện ích

- `explained_variance_ratio()` → tỉ lệ phương sai của từng eigenface ($\lambda_i / \sum \lambda$).
- `cumulative_explained_variance()` → phương sai tích lũy.
- `n_components_for_variance(0.95)` → cần bao nhiêu eigenfaces để đạt 95% phương sai.
- `print_summary()` → in tóm tắt mô hình.

---

### 8.3 `src/baseline.py` — Bước 3a: Phương pháp so sánh

**Mục đích:** Tạo phương pháp **đơn giản** (KNN trực tiếp trên pixel) để so sánh với Eigenfaces. Nếu Eigenfaces tốt hơn, ta chứng minh được phép chiếu vuông góc thực sự có ích.

**Class `PixelKNNRecognizer`:**

```python
class PixelKNNRecognizer:
    def __init__(self, k=1):
        self.k = k   # số láng giềng (KNN với k=1 là gần nhất)

    def fit(self, X_train, y_train):
        # KNN không "học" gì cả — chỉ lưu lại tập train
        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()

    def predict(self, X_test):
        # Với mỗi ảnh test, so sánh với toàn bộ train trong R^10304
        for i in range(N_test):
            diffs     = self.X_train_ - X_test[i]
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))
            k_nearest_idx    = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train_[k_nearest_idx]
            predictions[i]   = np.argmax(np.bincount(k_nearest_labels))  # bầu chọn đa số
```

**Khác biệt với Eigenfaces:**

| | Eigenfaces | Pixel-KNN |
|---|---|---|
| Không gian so sánh | $\mathbb{R}^{50}$ (sau chiếu) | $\mathbb{R}^{10304}$ (gốc) |
| Có bước "học" đặc trưng? | Có | Không |
| Tốc độ dự đoán | Nhanh | Chậm |
| Nhạy với ánh sáng, nhiễu? | Ít | Nhiều |
| Accuracy kỳ vọng | 92-97% | 80-88% |

---

### 8.4 `src/evaluator.py` — Bước 3b: Đánh giá và so sánh

**Các hàm chính:**

#### `accuracy(y_true, y_pred)`
$$\text{accuracy} = \frac{\text{số dự đoán đúng}}{\text{tổng số mẫu}}$$
```python
return float(np.mean(y_true == y_pred))
```

#### `per_class_accuracy(y_true, y_pred)`
Tính accuracy riêng cho từng người (để xem có người nào khó nhận dạng đặc biệt).

#### `compare_k_values(X_train, y_train, X_test, y_test, k_values)`
Huấn luyện Eigenfaces với nhiều giá trị $k$ khác nhau, đo accuracy + thời gian, tìm $k$ tối ưu.

```python
for k in [5, 10, 20, 30, 50, 75, 100, 150]:
    rec = OrthogonalFaceRecognizer(n_components=k).fit(X_train, y_train)
    y_pred = rec.predict(X_test)
    acc = accuracy(y_test, y_pred)
    # ghi nhận thời gian, accuracy
```

#### `run_full_comparison(...)`
- Chạy Eigenfaces với $k$ tốt nhất
- Chạy Baseline KNN
- In bảng so sánh chi tiết: accuracy, thời gian train, thời gian predict, hệ số nén, người khó nhận nhất

---

### 8.5 `src/visualizer.py` — Bước 4: Trực quan hóa

**Mục đích:** Sinh **5 biểu đồ** chuẩn cho báo cáo và slide. Tất cả lưu vào `outputs/`.

#### Hình 1: `mean_face.png` — Khuôn mặt trung bình

```python
def plot_mean_face(recognizer, save_path):
    mean_img = recognizer.mean_face_.reshape(112, 92)   # vector → ảnh 2D
    plt.imshow(mean_img, cmap="gray", vmin=0, vmax=255)
```

**Sẽ thấy gì:** Một khuôn mặt "mờ ảo" — đó là trung bình của 320 khuôn mặt. Bạn thoáng thấy được hình dạng đầu, mắt, mũi, miệng.

#### Hình 2: `eigenfaces.png` — Lưới top-20 eigenfaces

Mỗi eigenface khi reshape về 2D trông như một "khuôn mặt ma" — có cả vùng sáng và vùng tối. Hãy nhớ:
- Eigenfaces có giá trị âm → cần normalize về `[0, 1]` để hiển thị
- Eigenface #1 thường biểu diễn "ánh sáng tổng thể"
- Các eigenfaces sau biểu diễn các đặc trưng tinh tế hơn (mũi, mắt, kính, râu...)
- % phương sai mỗi eigenface giải thích được ghi dưới mỗi ảnh

#### Hình 3: `recognition.png` — Ví dụ nhận dạng đúng và sai

Hiển thị 1 trường hợp đúng (viền xanh) và 1 trường hợp sai (viền đỏ). Mỗi trường hợp gồm:
- Ảnh test (truy vấn)
- Ảnh khớp gần nhất trong train
- Ảnh đúng (chỉ cho trường hợp sai)

#### Hình 4: `accuracy_vs_k.png` — Accuracy theo $k$

Đường cong accuracy theo $k$ (5, 10, 20, 30, 50, 75, 100, 150). Có đánh dấu $k$ tốt nhất và đường ngang biểu thị accuracy của baseline.

**Sẽ thấy gì:**
- $k$ quá nhỏ → accuracy thấp (thiếu thông tin)
- $k$ tăng dần → accuracy tăng nhanh
- Vượt qua "sweet spot" → accuracy bão hòa hoặc giảm nhẹ (do overfitting/nhiễu)
- Eigenfaces luôn cao hơn đường baseline

#### Hình 5: `variance_ratio.png` — Phương sai giải thích

Biểu đồ kép:
- Cột xanh: % phương sai từng eigenface
- Đường cam: phương sai tích lũy
- Đường đỏ: ngưỡng 95% và $k$ tương ứng

**Sẽ thấy gì:** Eigenface đầu tiên chiếm phần lớn phương sai (~15-20%), giảm dần. Khoảng 30-50 eigenfaces là đủ đạt 95% phương sai.

---

### 8.6 `manual_example.py` — Bước 5: Ví dụ tính tay

**Mục đích:** Đây là phần **cực kỳ quan trọng cho báo cáo** — bạn chứng minh được mình hiểu thuật toán bằng cách **tính tay** trên dữ liệu siêu nhỏ.

**Dữ liệu đồ chơi:**
- 4 ảnh 3×3 pixel (mỗi ảnh là vector 9 chiều)
- Người 1: 2 ảnh "tối" (gradient tăng từ trái sang phải)
- Người 2: 2 ảnh "sáng" (gradient giảm từ trái sang phải)
- 1 ảnh test gần với Người 1

**File chạy qua 9 bước, in từng phép tính ra console:**

| Bước | In ra |
|---|---|
| 1 | Hiển thị 4 ảnh train + ảnh test |
| 2 | Tính $\bar{\mathbf{x}}$ — in từng phép cộng từng pixel: $(10+12+90+88)/4 = 50$, ... |
| 3 | Tính $\Phi_i = \mathbf{x}_i - \bar{\mathbf{x}}$ cho từng ảnh |
| 4 | Tính ma trận $L = \Phi \Phi^T / N$ — in từng phần tử $L_{ij}$ |
| 5 | Phân rã trị riêng của $L$ — in eigenvalues gốc, sau khi sắp xếp giảm dần, % phương sai |
| 6 | Khôi phục $\mathbf{u}_i = \Phi^T \mathbf{v}_i$, chuẩn hóa, kiểm tra $U^T U = I$ |
| 7 | Chiếu mỗi ảnh train → in $\hat{\mathbf{x}}_i$ |
| 8 | Chiếu ảnh test, tính khoảng cách tới mỗi ảnh train, kết luận |
| 9 | Vẽ biểu đồ: ảnh train + eigenfaces + scatter plot 2D |

**Output cuối:** `outputs/manual_example.png` — biểu đồ minh họa.

**Quan trọng:** Khi báo cáo, bạn có thể **copy thẳng output của file này** vào bài báo cáo để chứng minh thuật toán hoạt động đúng. Có người chấm điểm sẽ kiểm tra bằng cách tính lại theo công thức.

---

### 8.7 `extended_applications.py` — Bước 6: Hai ứng dụng mở rộng

**Mục đích:** Đáp ứng yêu cầu của đề bài: *"ứng dụng trong nhiều lĩnh vực sẽ được đánh giá cao hơn"*.

#### Hàm metric: `mse` và `psnr`

**MSE (Mean Squared Error):**
$$\text{MSE} = \frac{1}{p}\sum_{i=1}^{p}(\hat{x}_i - x_i)^2$$
- Càng nhỏ càng tốt. MSE = 0 → tái tạo hoàn hảo.
- Đơn vị: bình phương đơn vị pixel.

**PSNR (Peak Signal-to-Noise Ratio):**
$$\text{PSNR} = 10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \text{ [dB]}$$
- Càng lớn càng tốt. PSNR = ∞ → tái tạo hoàn hảo.
- Quy ước thực tế:

| PSNR | Chất lượng |
|---|---|
| > 40 dB | Rất tốt — khó phân biệt với gốc |
| 30-40 dB | Tốt |
| 20-30 dB | Chấp nhận được |
| < 20 dB | Kém — sai khác rõ |

#### Ứng dụng 1: Tái tạo ảnh

**Mục đích:** Nén ảnh — chỉ lưu $k$ tọa độ eigenface thay vì 10304 pixel.

**Công thức:**
$$\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

**Hàm `plot_reconstruction_comparison(...)`:**
- Chọn 1 ảnh test
- Tái tạo với $k = 1, 5, 20, 50, 100, 150$
- Vẽ thành 1 hàng: gốc | k=1 | k=5 | ... | k=150
- Mỗi ảnh ghi MSE và PSNR
- Lưu vào `app1_reconstruction.png`

**Hàm `plot_reconstruction_quality(...)`:**
- Vẽ 2 đường cong: MSE theo $k$ (giảm) và PSNR theo $k$ (tăng)
- Có đường tham chiếu PSNR = 30 dB và 40 dB
- Lưu vào `app1_quality_curve.png`

**Kết quả kỳ vọng:**

| $k$ | PSNR | Diện mạo |
|---|---|---|
| 1 | ~3 dB | Chỉ thấy ánh sáng tổng thể |
| 5 | ~10 dB | Mơ hồ nhận ra khuôn mặt |
| 20 | ~25 dB | Nhận ra rõ, có chi tiết |
| 50 | ~35 dB | Khá tốt |
| 100 | ~42 dB | Rất tốt |
| 150 | ~48 dB | Gần như hoàn hảo |

**Ý nghĩa:** Chỉ với 50 con số (thay vì 10304 pixel), ta đã có thể tái tạo ảnh chất lượng cao → **hệ số nén ~200x**.

#### Ứng dụng 2: Làm mờ ảnh

**Mục đích:** Làm mờ ảnh **bằng cách dùng phép chiếu vuông góc** (không dùng filter Gaussian truyền thống).

**Ý tưởng:**
- Eigenfaces đầu tiên (eigenvalue LỚN) nắm bắt các thành phần **tần số thấp** — cấu trúc tổng thể (hình dạng khuôn mặt, vùng sáng/tối lớn)
- Eigenfaces sau (eigenvalue NHỎ) nắm bắt **tần số cao** — chi tiết nhỏ, cạnh, nếp nhăn
- Tái tạo chỉ với $k$ eigenfaces đầu = **bỏ tần số cao** = **bộ lọc thông thấp** = **làm mờ ảnh**
- $k$ càng nhỏ → ảnh càng mờ

**Công thức:** Vẫn là công thức tái tạo ở trên! Nhưng dùng $k$ nhỏ.

**Hàm `plot_blurring_effect(...)`:**
- Dùng $k = 1, 3, 7, 15, 30$ (nhỏ để thấy rõ mờ)
- Vẽ thành 1 hàng: gốc | k=1 | k=3 | k=7 | k=15 | k=30
- Lưu vào `app2_blurring.png`

**So sánh với phương pháp truyền thống:**

| | Bộ lọc Gaussian | Phép chiếu eigenface |
|---|---|---|
| Cơ chế | Trung bình hóa pixel xung quanh | Bỏ thành phần tần số cao |
| Yếu tố điều chỉnh | Độ rộng kernel (σ) | Số eigenfaces ($k$) |
| Cần huấn luyện trước? | Không | Có (cần học eigenfaces từ tập train) |
| Cơ sở toán học | Tích chập (convolution) | Phép chiếu vuông góc |

**Lý do dùng phép chiếu thay vì Gaussian:** Để cho thấy phép chiếu vuông góc có thể làm được hiệu ứng làm mờ — chính là yêu cầu của đề bài.

#### Hàm `run_extended_applications(...)`

Gói cả hai ứng dụng lại, chạy lần lượt và sinh tất cả biểu đồ.

---

### 8.8 `main_projection.py` — Entry point

File này **không chứa thuật toán** — chỉ điều phối toàn bộ pipeline.

```python
def main():
    # Bước 1: Tải + chia dữ liệu
    X_train, y_train, X_test, y_test = load_orl_dataset(...)

    # Bước 3a: Tìm k tốt nhất
    k_results = compare_k_values(...)
    best_k = k_results["best_k"]

    # Bước 2: Huấn luyện với k tốt nhất
    recognizer = OrthogonalFaceRecognizer(n_components=best_k).fit(X_train, y_train)

    # Bước 3b: So sánh đầy đủ với baseline
    comparison = run_full_comparison(...)

    # Bước 4: Sinh 5 biểu đồ
    plot_all(...)

    # Bước 6: Huấn luyện lại với k=150 (cần đủ eigenfaces cho tái tạo)
    rec_extended = OrthogonalFaceRecognizer(n_components=150).fit(X_train, y_train)
    run_extended_applications(rec_extended, ...)
```

**Tại sao Bước 6 phải huấn luyện lại với $k$ lớn?** Vì ở Bước 6 ta thử nhiều giá trị $k$ khác nhau (lên đến 150). Nếu mô hình chỉ có 50 eigenfaces (best_k) thì không đủ cho thí nghiệm $k=150$.

---

## 9. BA ỨNG DỤNG — HIỂU SÂU

### 9.1 Ứng dụng 1: Nhận dạng khuôn mặt

**Bài toán:** Cho ảnh khuôn mặt mới, máy trả lời "đây là ai trong 40 người?".

**Cách phép chiếu vuông góc giúp:**

1. **Giảm chiều mạnh** (10304 → 50): so sánh nhanh hơn 200 lần.
2. **Tăng độ chính xác**: trong không gian gốc, mỗi pixel có vai trò bằng nhau và bị ảnh hưởng bởi ánh sáng/nhiễu. Trong không gian eigenface, các tọa độ là "đặc trưng có ý nghĩa".
3. **Phép chiếu vuông góc là phép xấp xỉ tốt nhất** (theo nghĩa MSE) — đảm bảo $\hat{\mathbf{y}}$ giữ được phần lớn thông tin về $\mathbf{y}$.

**Khoảng cách Euclidean được dùng:**
$$d(\mathbf{y}_1, \mathbf{y}_2) = \|\hat{\mathbf{y}}_1 - \hat{\mathbf{y}}_2\|_2$$

**Tính chất quan trọng:** Phép chiếu vuông góc giữ nguyên hoặc rút ngắn khoảng cách:
$$\|\hat{\mathbf{y}}_1 - \hat{\mathbf{y}}_2\| \leq \|\mathbf{y}_1 - \mathbf{y}_2\|$$
(Bất đẳng thức tam giác trong không gian Hilbert.)

### 9.2 Ứng dụng 2: Tái tạo ảnh (nén)

**Bài toán:** Nén ảnh xuống cực nhỏ rồi giải nén — chấp nhận một chút mất mát.

**Cách phép chiếu vuông góc giúp:**

- Lưu trữ: chỉ cần $k$ con số (~50) thay vì 10304 pixel.
- Tái tạo: chiếu ngược lại $\hat{\mathbf{x}} = U_k \hat{\mathbf{y}} + \bar{\mathbf{x}}$.
- Sai số tái tạo $\|\mathbf{x} - \hat{\mathbf{x}}\|$ là **nhỏ nhất có thể** với $k$ chiều — đó là tính chất "best approximation" của phép chiếu vuông góc.

**Liên hệ với SVD và PCA:**
- Phương pháp này tương đương với **PCA** (Principal Component Analysis).
- Cũng tương đương với **truncated SVD** của ma trận dữ liệu.
- Eigenfaces chính là các **principal components** trong không gian ảnh.

### 9.3 Ứng dụng 3: Làm mờ ảnh (low-pass filter)

**Bài toán:** Làm mờ ảnh để bỏ chi tiết nhỏ (như áp dụng filter mờ trên Instagram).

**Cách phép chiếu vuông góc giúp:**

Eigenfaces được sắp xếp theo eigenvalue giảm dần. Có một quan sát quan trọng:

> Các eigenfaces ứng với eigenvalue LỚN nắm bắt thông tin **tần số thấp** (cấu trúc tổng thể).
> Các eigenfaces ứng với eigenvalue NHỎ nắm bắt thông tin **tần số cao** (chi tiết, cạnh).

→ Tái tạo với $k$ nhỏ = giữ tần số thấp, bỏ tần số cao = làm mờ.

**Liên hệ với xử lý tín hiệu:**
- Trong xử lý tín hiệu, "tần số thấp" tương ứng với biến đổi chậm trong không gian (mảng sáng đều, vùng lớn).
- "Tần số cao" tương ứng với biến đổi nhanh (cạnh, đốm, nhiễu).
- Bộ lọc Gaussian/box filter truyền thống cũng có tác dụng tương tự, nhưng dùng tích chập (convolution).
- Phương pháp này dùng phép chiếu xuống không gian con eigenface — tương đương với một bộ lọc thông thấp **được thiết kế theo dữ liệu** (data-driven low-pass filter).

---

## 10. KẾT QUẢ KỲ VỌNG VÀ CÁCH ĐỌC BIỂU ĐỒ

### 10.1 Bảng kết quả nhận dạng

| Phương pháp | Accuracy kỳ vọng | Giải thích |
|---|---|---|
| Eigenfaces ($k$ ≈ 50) | 92-97% | Mạnh, ổn định |
| Pixel-KNN (baseline) | 80-88% | Yếu hơn, chậm hơn |
| Chênh lệch | +5-15% | Eigenfaces vượt trội |

### 10.2 Bảng chất lượng tái tạo

| $k$ eigenfaces | PSNR | Đánh giá |
|---|---|---|
| 1 | ~3 dB | Cực kém |
| 5 | ~10 dB | Kém |
| 20 | ~25 dB | Chấp nhận được |
| 50 | ~35 dB | Tốt |
| 100 | ~42 dB | Rất tốt |
| 150 | ~48 dB | Gần như hoàn hảo |

### 10.3 Cách đọc 8 biểu đồ

| File | Mục đích | Quan sát chính |
|---|---|---|
| `mean_face.png` | Khuôn mặt trung bình | Hình "trung bình" của 320 ảnh — mờ ảo |
| `eigenfaces.png` | Top-20 eigenfaces | Mỗi ảnh là "1 hướng biến thiên" |
| `recognition.png` | Ví dụ nhận dạng | 1 ĐÚNG (xanh) + 1 SAI (đỏ) |
| `accuracy_vs_k.png` | Accuracy theo $k$ | Sweet spot ~50, Eigenfaces > baseline |
| `variance_ratio.png` | Phương sai từng eigenface | EF#1 chiếm nhiều nhất, ~30-50 EF đạt 95% |
| `app1_reconstruction.png` | Tái tạo với $k$ khác nhau | $k$ lớn → ảnh sắc nét |
| `app1_quality_curve.png` | MSE/PSNR theo $k$ | Đường cong giảm/tăng nhanh đầu, bão hòa sau |
| `app2_blurring.png` | Hiệu ứng làm mờ | $k$ nhỏ → ảnh mờ rõ |

---

## 11. CÂU HỎI THƯỜNG GẶP KHI BÁO CÁO

### Q1: Phép chiếu vuông góc trong bài là gì? Công thức?

**A:** Là phép chiếu vector ảnh $\mathbf{x}$ xuống không gian con sinh bởi các eigenfaces.

Công thức:
- Tọa độ: $\hat{\mathbf{y}} = U_k^T (\mathbf{x} - \bar{\mathbf{x}})$
- Hình chiếu: $\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$

Gọi là "vuông góc" vì các cột của $U_k$ tạo cơ sở **trực chuẩn** ($U_k^T U_k = I$).

### Q2: Tại sao không tính $C$ trực tiếp mà phải dùng thủ thuật $L$?

**A:** Vì $C$ có kích thước $10304 \times 10304$ — quá lớn:
- Chiếm ~800 MB RAM
- Phân rã trị riêng cực chậm (hàng giờ)

$L = \Phi \Phi^T / N$ chỉ có kích thước $320 \times 320$ — nhanh hơn hàng triệu lần.

Hai ma trận có **cùng eigenvalue khác 0**. Eigenvector của $C$ được khôi phục bằng $\mathbf{u} = \Phi^T \mathbf{v}$ (đã chứng minh ở mục 5.2).

### Q3: Tại sao dùng `np.linalg.eigh()` thay vì `eig()`?

**A:** Vì $L$ là ma trận **đối xứng thực**:
- `eigh()` được tối ưu cho ma trận đối xứng → ổn định số học hơn
- Đảm bảo trị riêng là **số thực** (không có phần ảo do sai số)
- Trả về eigenvectors **trực chuẩn** sẵn

### Q4: Vai trò của $\bar{\mathbf{x}}$ (mean face) là gì?

**A:** Trung tâm hóa dữ liệu trước khi tính hiệp phương sai. Nếu không trừ $\bar{\mathbf{x}}$:
- Ma trận hiệp phương sai sẽ "nghiêng" về hướng trung bình
- Eigenfaces đầu tiên sẽ chủ yếu phản ánh "có khuôn mặt" thay vì "khác biệt giữa các khuôn mặt"

Lúc nhận dạng cũng phải trừ $\bar{\mathbf{x}}$ ra khỏi ảnh test để giữ tính nhất quán.

### Q5: Sao chọn $k = 50$? Sao không nhỏ hơn hay lớn hơn?

**A:** Có 2 cách chọn:
1. **Theo phương sai:** Cần $k$ sao cho top-$k$ giải thích ≥ 95% phương sai. Trong dữ liệu này, $k \approx 30-50$ là đủ.
2. **Theo accuracy:** Thử $k = 5, 10, 20, 30, 50, 75, 100, 150$, chọn $k$ cho accuracy cao nhất trên tập test.

Trong dự án, ta dùng cách 2 (chính xác hơn cho bài toán nhận dạng) trong `compare_k_values()`.

$k$ quá lớn: thêm eigenfaces sẽ học các thành phần phương sai nhỏ — thường là **nhiễu** → accuracy có thể GIẢM.

### Q6: Tại sao Eigenfaces tốt hơn Pixel-KNN?

**A:** Có 3 lý do:
1. **Bỏ nhiễu:** Eigenfaces giữ thông tin "có ý nghĩa", bỏ thông tin nhiễu.
2. **Bất biến một phần với ánh sáng:** Eigenface #1 thường mã hóa "độ sáng tổng thể" → ảnh cùng người dù sáng/tối vẫn gần nhau trong không gian eigenface (sau khi bỏ EF#1 nếu cần).
3. **So sánh trong không gian thấp chiều:** Khoảng cách Euclidean trong $\mathbb{R}^{10304}$ không ý nghĩa lắm (mọi điểm gần như cách đều nhau — "curse of dimensionality"). Trong $\mathbb{R}^{50}$ khoảng cách có ý nghĩa hơn.

### Q7: Vì sao tái tạo và làm mờ dùng CÙNG công thức?

**A:** Cả hai đều dùng:
$$\hat{\mathbf{x}} = U_k U_k^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}$$

Khác biệt **chỉ ở số $k$**:
- **Tái tạo:** $k$ lớn (50-150) → giữ nhiều thông tin → ảnh sắc nét
- **Làm mờ:** $k$ nhỏ (1-30) → giữ ít thông tin → ảnh mờ

Đây là vẻ đẹp toán học: cùng một công thức cho hai mục đích đối lập, chỉ phụ thuộc ngưỡng cắt.

### Q8: Sai số tái tạo có ý nghĩa gì?

**A:** Sai số $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$ là **thành phần của $\mathbf{x} - \bar{\mathbf{x}}$ vuông góc với không gian eigenface**.

Tính chất:
- $\mathbf{e}^T U_k = \mathbf{0}$ (vuông góc với mọi eigenface đã giữ)
- $\|\mathbf{e}\|$ là sai số nhỏ nhất có thể với $k$ eigenfaces (best approximation)
- $\|\mathbf{e}\|^2 = \sum_{i=k+1}^{p} \lambda_i$ (tổng các eigenvalues đã bỏ)

### Q9: Tại sao chọn 1-NN (Nearest Neighbor) thay vì k-NN hay phương pháp khác?

**A:** Vì:
- Dataset ORL nhỏ (8 ảnh/người) → k-NN với $k > 1$ có thể "bầu" nhầm.
- 1-NN đơn giản, không có siêu tham số — dễ hiểu và giải thích.
- Trong không gian eigenface đã được giảm chiều, 1-NN đủ chính xác.

### Q10: Có thể dùng SVD thay vì eigendecomposition không?

**A:** Hoàn toàn được. Thực tế **cả hai cách tương đương**:

SVD của $\Phi$: $\Phi = U \Sigma V^T$
- Các cột của $U$ chính là **eigenfaces** (trị riêng = $\sigma^2 / N$)
- Trong dự án, ta dùng eigendecomposition vì:
  - Phù hợp với cách trình bày toán học truyền thống của Eigenfaces (1991)
  - Cho phép áp dụng thủ thuật Turk & Pentland để giảm chi phí tính toán

---

## 12. TÀI LIỆU THAM KHẢO

```
[1] Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition."
    Journal of Cognitive Neuroscience, 3(1), 71-86.
    → Bài báo gốc của thuật toán Eigenfaces. Đề xuất thủ thuật L thay vì C.

[2] Jolliffe, I. T. (2002). "Principal Component Analysis." 2nd ed. Springer.
    → Sách kinh điển về PCA — nền tảng toán học của Eigenfaces.

[3] Gonzalez, R. C. & Woods, R. E. (2018). "Digital Image Processing." 4th ed. Pearson.
    → Định nghĩa MSE, PSNR, bộ lọc thông thấp trong xử lý ảnh số.

[4] Strang, G. (2016). "Introduction to Linear Algebra." 5th ed. Wellesley-Cambridge Press.
    → Cơ sở lý thuyết về phép chiếu vuông góc, eigendecomposition, không gian con.

[5] Cover, T. & Hart, P. (1967). "Nearest Neighbor Pattern Classification."
    IEEE Transactions on Information Theory, 13(1), 21-27.
    → Bài báo gốc về Nearest Neighbor classifier.

[6] AT&T Laboratories Cambridge. "The ORL Database of Faces."
    https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    → Nguồn dataset gốc.

[7] NumPy documentation — numpy.linalg.eigh.
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html

[8] Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment."
    Computing in Science & Engineering, 9(3), 90-95.
```

---

## TÓM TẮT NHANH (CHEAT SHEET CHO BÁO CÁO)

```
┌─────────────────────────────────────────────────────────────────┐
│   THUẬT TOÁN EIGENFACES (TURK & PENTLAND, 1991)                 │
│   Một ứng dụng kinh điển của PHÉP CHIẾU VUÔNG GÓC               │
└─────────────────────────────────────────────────────────────────┘

INPUT  : N ảnh huấn luyện (mỗi ảnh là vector p chiều)
OUTPUT : Có thể nhận dạng, nén, làm mờ ảnh

──── BƯỚC 1: TIỀN XỬ LÝ ────────────────────────────────────────
   Mean face       : x̄ = (1/N) Σ xᵢ
   Trung tâm hóa   : Φᵢ = xᵢ - x̄

──── BƯỚC 2: HỌC KHÔNG GIAN EIGENFACE ──────────────────────────
   Thủ thuật       : L = (1/N) Φ Φᵀ      (kích thước N×N, NHỎ)
                     C = (1/N) Φᵀ Φ      (kích thước p×p, LỚN)
   Phân rã         : L·v = λ·v           (eigh)
   Khôi phục       : uᵢ = Φᵀvᵢ / ‖Φᵀvᵢ‖  (eigenfaces)
   Chọn k          : U_k = [u₁ | u₂ | ... | u_k]

──── BƯỚC 3: PHÉP CHIẾU VUÔNG GÓC ──────────────────────────────
   Tọa độ          : ŷ = U_k^T (y - x̄)
   Hình chiếu      : ŷ_proj = U_k U_k^T (y - x̄) + x̄

──── BƯỚC 4: ỨNG DỤNG ─────────────────────────────────────────
   1. Nhận dạng    : argmin_i ‖ŷ - ŷᵢ_train‖     (1-NN)
   2. Tái tạo      : x̂ = U_k U_k^T (x - x̄) + x̄  (k LỚN, ~50-150)
   3. Làm mờ       : x̂ = U_k U_k^T (x - x̄) + x̄  (k NHỎ, ~1-30)

──── ĐO LƯỜNG ─────────────────────────────────────────────────
   Accuracy        = số đúng / tổng số
   MSE             = (1/p) Σ (x̂ᵢ - xᵢ)²
   PSNR            = 10 · log₁₀(255² / MSE)  [dB]

──── KẾT QUẢ KỲ VỌNG ──────────────────────────────────────────
   Eigenfaces (k=50)   : Accuracy 92-97%
   Pixel-KNN baseline  : Accuracy 80-88%
   Cải thiện           : +5-15%
   Nén                 : 10304 → 50 chiều  (200x)
```

---

> **Lời khuyên cuối:** Khi đi báo cáo, hãy **bắt đầu từ trực giác** (phép chiếu vuông góc trong 2D, 3D) rồi mới đi vào công thức. Đừng "đọc thuộc" — hãy hiểu **tại sao** mỗi bước được thiết kế như vậy. Người chấm sẽ hỏi tại sao chứ không hỏi "công thức là gì".

> **Chúc thành công với bài tập lớn!**
