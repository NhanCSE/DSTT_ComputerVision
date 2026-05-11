# BÀI TẬP LỚN ĐẠI SỐ TUYẾN TÍNH

**ĐỀ TÀI ĐƯỢC PHÂN CÔNG:**
1. Nghiên cứu tính chất và ứng dụng của phép chiếu vuông góc trong thị giác máy tính.
2. Ứng dụng: Nhận dạng đối tượng, tái tạo hình ảnh và kỹ thuật làm mờ hình ảnh.

---

## PHẦN 1. QUY ĐỊNH VỀ BÀI TẬP LỚN

### I. CÁCH THỰC HIỆN
1. Hai nhóm thực hiện 1 đề tài theo sự phân công của giảng viên.
2. Mỗi nhóm chuẩn bị một bài báo cáo theo yêu cầu của đề và slide thuyết trình, ít nhất 3 câu hỏi chuẩn bị phản biện.
3. Bài báo cáo và slide thuyết trình nộp trước 2 ngày so với buổi thuyết trình qua email: 010037@tmp.hcmut.edu.vn và nộp bản in báo cáo vào ngày cuối cùng của buổi học (04/06/2026).
4. Trong buổi báo cáo, 2 nhóm sẽ bốc thăm để xác định nhóm thuyết trình/nhóm phản biện.

### II. CÁCH ĐÁNH GIÁ

| Tiêu chí | % điểm Bài tập lớn | Tiêu chí con |
| :--- | :--- | :--- |
| **Nộp bài và điểm danh** | 10% | • Nộp bài (50%)<br>• Điểm danh (50%) |
| **Bài báo cáo** | 50% | • Giới thiệu thông tin môn học và nhóm; nêu đối tượng, mục tiêu và phương pháp của đề tài (20%)<br>• Trình bày nội dung và phương pháp (20%)<br>• Cơ sở lý thuyết của đề tài và tính chặt chẽ (20%)<br>• Thuật toán và code và kết quả của đề tài (20%)<br>• Viết và trình bày văn bản đầy đủ, đúng chuẩn (20%)<br>1. Cỡ chữ 13-17. Lề trái 3cm; còn lại 2cm.<br>2. Nội dung tối thiểu 20 trang.<br>3. không chụp hình công thức toán.<br>4. Đầy đủ cấu trúc bài báo cáo: Thông tin môn học và nhóm; nêu đối tượng, mục tiêu và phương pháp của đề tài; cơ sở lý thuyết; thuật toán; ví dụ; phần mềm; kết quả đạt được; tài liệu tham khảo.<br>5. Trang bìa đủ thông tin: Đề tài; khoa; môn học; lớp; nhóm; giảng viên hướng dẫn; thành viên nhóm; mã số sinh viên; mô tả đóng góp |
| **Hỏi đáp** | 40% | • Đối tượng và kết quả của đề tài (20%).<br>• Nội dung phương pháp (20%).<br>• Cơ sở lý thuyết (20%).<br>• Thuật toán và code (20%).<br>• Kết quả và mở rộng (20%). |

> **Ghi chú:** Nếu sinh viên vắng báo cáo: tối đa là 50% điểm.

---

### III. TIÊU CHÍ ĐÁNH GIÁ BÀI TẬP LỚN

Yêu cầu đối với một đề tài bài tập lớn:

**1. Lý thuyết**
* Ngắn gọn vừa đủ để phục vụ cho đề tài, giới hạn khoảng 4 đến 10 trang A4.
* Những phần lý thuyết đã được học trong môn ĐSTT không chép lại.
* Kiến thức lấy từ nguồn nào thì phải trích dẫn rõ ràng, khuyến khích lấy tài liệu từ các nhà xuất bản uy tín.
* Ví dụ định nghĩa phân tích SVD được trích dẫn từ [1], trong đó [1] là cuốn sách đã được liệt kê trong phần tài liệu.

**2. Thực hành**
* Nêu rõ các bước để giải quyết bài toán (nếu có thuật toán thì càng tốt).
* Có ví dụ minh họa cho thuật toán.
* Có ít nhất một bài toán thực tế. Nếu có các ứng dụng trong nhiều lĩnh vực khác nhau sẽ được đánh giá cao hơn.
* Ứng dụng ở đây không tính là ứng dụng vấn đề trong đề tài để giải hoặc chứng minh các bài toán khác. Ví dụ ứng dụng của định thức để tính thể tích hình tứ diện không được tính là ứng dụng, ứng dụng của dạng toàn phương trong bài toán cực trị cũng không được tính,...
* Ví dụ một ứng dụng: dùng khai triển Fourier rời rạc để khử nhiễu. Nêu các bước giải quyết bài toán.
    * **Bước 1.** Từ file âm thanh, dùng matlab số hóa file, dùng Fourier để phân tích sang miền tần số.
    * **Bước 2.** Khử nhiễu. Phải hiểu tại sao làm như vậy ta khử được nhiễu.
    * **Bước 3.** Khôi phục lại âm thanh.
* Sinh viên có thể lấy một file âm thanh có sẵn trên mạng, hoặc tự ghi âm file âm thanh một bài hát có lẫn nhiễu như tiếng xe, tiếng chó sủa.
* Sau khi khử file thì mất nhiễu. Không khuyến khích dùng matlab để giả lập nhiễu.
* Đề tài sẽ được cho điểm cao nếu có so sánh các phương pháp khác nhau để giải quyết bài toán.

**3. Coding**
* Sinh viên năm nhất chưa có kinh nghiệm trong lập trình nên không bắt buộc phải viết code chi tiết, không dùng các hàm có sẵn.
* Tuy nhiên code đã có nhiều trên mạng nên sinh viên có thể lấy về, phải ghi nguồn và hiểu rõ từng dòng code.
* Có thể viết code bằng các ngôn ngữ hoặc phần mềm khác nhau (Python, matlab,...) có thể nhờ ChatGPT để viết, quan trọng là sinh viên phải hiểu rõ đoạn code (nếu có).

---

### IV. MỘT SỐ LƯU Ý
* Đề tài có thể được viết bằng file word, pdf, ppt đều được.
* Trang đầu tiên phải ghi tên đề tài, danh sách nhóm gồm mssv, họ và tên (lưu ý ghi cẩn thận, không để sai tên đầy đủ).
* Một đề tài phải có các phần: Lý thuyết, thực hành, kết luận, tài liệu tham khảo (các nguồn trên mạng phải ghi link).
* Trước ngày báo cáo khoảng 2 ngày sinh viên phải gởi mail file (word, pdf) và file trình chiếu ppt cho giảng viên qua email: 010037@tmp.hcmut.edu.vn.
* Phải có mặt trong lúc báo cáo. Nếu vắng mặt phải có lý do và được lấy điểm thấp hơn.
* Chuẩn bị file powerpoint và chọn một người (hoặc giảng viên chọn ngẫu nhiên) để thuyết trình khoảng 10 phút: ngắn gọn lý thuyết, bài toán thực tế, code.
* Bài tập lớn được thực hiện trên tinh thần làm việc nhóm (team work) nên các thành viên trong nhóm phải có trách nhiệm trao đổi, lập nhóm ngay từ khi được giao đề tài, cử nhóm trưởng, phân công nhiệm vụ của từng người, thường xuyên trao đổi lẫn nhau, giúp các bạn trong nhóm hiểu được vấn đề.
* Hình thức báo cáo: trực tiếp trên lớp.
* Thời gian báo cáo: Cô sẽ thông báo trên lớp.

### V. YÊU CẦU CHUNG
1. Lấy ví dụ ngoài sách giáo trình.
2. BÀI BÁO CÁO: Đóng thành tập, trang đầu tiên có bảng Danh sách sinh viên gồm: số thứ tự, tên sinh viên, MSSV, tên lớp, Đề tài Bài tập lớn.
3. Chỉ được viết những gì tất cả các thành viên trong nhóm hiểu: GV sẽ hỏi ngẫu nhiên 1 SV trong nhóm, nếu không trả lời được thì cả nhóm bị trừ điểm.
4. Copy tài liệu hay đoạn code thì phải ghi rõ trích dẫn (Chỉ copy sau khi cả nhóm hiểu).
5. Dùng phần mềm Matlab, Python,... để chạy ví dụ.