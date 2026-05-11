"""
Bước 1: Chuẩn bị Dataset & Dataloader
======================================
Tập dữ liệu: AT&T Database of Faces (còn gọi là ORL Faces Database)
  - 40 người, 10 ảnh/người → 400 ảnh tổng cộng
  - Kích thước mỗi ảnh: 92 x 112 pixel, grayscale

Cấu trúc thư mục sau khi giải nén:
    data/orl_faces/
    ├── s1/       ← người số 1
    │   ├── 1.pgm
    │   ├── 2.pgm
    │   └── ...
    │   └── 10.pgm
    ├── s2/
    │   └── ...
    └── s40/

Tài liệu tham khảo:
  [1] AT&T Laboratories Cambridge – "The ORL Database of Faces"
      https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  [2] Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition."
      Journal of Cognitive Neuroscience, 3(1), 71–86.
"""

import os
import shutil
import urllib.request
import zipfile
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Hằng số kích thước ảnh gốc của ORL dataset
# ---------------------------------------------------------------------------
IMG_HEIGHT = 112   # chiều cao (pixel)
IMG_WIDTH  = 92    # chiều rộng (pixel)
IMG_SIZE   = IMG_HEIGHT * IMG_WIDTH  # = 10304 — kích thước vector 1D sau khi flatten

# ---------------------------------------------------------------------------
# URL tải dataset (mirror dự phòng nếu nguồn chính không hoạt động)
# ---------------------------------------------------------------------------
_ORL_PRIMARY_URL = (
    "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"
)
_ORL_FALLBACK_URL = (
    "https://github.com/rmilano24/orl-faces-dataset/raw/master/att_faces.zip"
)


# ===========================================================================
# HÀM 1: Tải dataset tự động
# ===========================================================================
def download_orl_dataset(save_dir: str = "data") -> str:
    """
    Tải và giải nén AT&T/ORL Faces dataset vào thư mục data/.

    Nếu dataset đã tồn tại, hàm này bỏ qua và trả về đường dẫn ngay.

    Tham số:
        save_dir (str): thư mục cha để lưu dataset.

    Trả về:
        str: đường dẫn tới thư mục orl_faces/ (chứa các thư mục s1..s40).
    """
    orl_dir  = os.path.join(save_dir, "orl_faces")
    zip_path = os.path.join(save_dir, "att_faces.zip")

    # Nếu đã giải nén sẵn thì không cần tải lại
    if os.path.isdir(orl_dir) and os.listdir(orl_dir):
        print(f"[OK] Dataset đã có tại: {orl_dir}")
        return orl_dir

    os.makedirs(save_dir, exist_ok=True)

    # Thử tải từ nguồn chính, nếu lỗi thì thử nguồn dự phòng
    for url in [_ORL_PRIMARY_URL, _ORL_FALLBACK_URL]:
        try:
            print(f"Đang tải dataset từ:\n  {url}")
            urllib.request.urlretrieve(url, zip_path)
            print("[OK] Tải xong. Đang giải nén...")
            break
        except Exception as e:
            print(f"[WARN] Không thể tải từ {url}: {e}")
    else:
        # Cả hai URL đều thất bại → in hướng dẫn thủ công
        _print_manual_download_guide(orl_dir)
        raise FileNotFoundError(
            "Không tải được dataset tự động. Xem hướng dẫn tải thủ công ở trên."
        )

    # Giải nén file zip
    # Zip chứa cấu trúc: att_faces/s1/, att_faces/s2/, ..., att_faces/s40/
    # Chúng ta muốn: data/orl_faces/s1/, data/orl_faces/s2/, ...
    
    # Giải nén vào thư mục tạm thời trước
    temp_dir = os.path.join(save_dir, "_temp_extract")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    os.remove(zip_path)
    
    # Tìm thư mục att_faces (hoặc tên khác nếu khác) trong thư mục tạm
    subdirs = os.listdir(temp_dir)
    src_dir = None
    for subdir in subdirs:
        full_path = os.path.join(temp_dir, subdir)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "s1")):
            src_dir = full_path
            break
    
    # Nếu không tìm thấy, giả sử s1..s40 ở trong temp_dir trực tiếp
    if src_dir is None:
        src_dir = temp_dir
    
    # Xóa orl_dir nếu đã tồn tại, rồi di chuyển src_dir → orl_dir
    if os.path.isdir(orl_dir):
        shutil.rmtree(orl_dir)
    os.rename(src_dir, orl_dir)
    
    # Xóa thư mục tạm nếu nó khác với orl_dir
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"[OK] Giải nén xong. Dataset tại: {orl_dir}")
    return orl_dir


def _print_manual_download_guide(orl_dir: str) -> None:
    """In hướng dẫn tải thủ công khi tải tự động thất bại."""
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN TẢI DATASET THỦ CÔNG")
    print("=" * 60)
    print("1. Truy cập link:")
    print("   https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip")
    print("   (hoặc tìm 'ORL Face Database download' trên Google)")
    print(f"2. Giải nén vào thư mục: {orl_dir}")
    print("   Cấu trúc thư mục phải là:")
    print(f"   {orl_dir}/s1/1.pgm, 2.pgm, ..., 10.pgm")
    print(f"   {orl_dir}/s2/1.pgm, ...")
    print(f"   ...  (cho đến s40)")
    print("=" * 60 + "\n")


# ===========================================================================
# HÀM 2: Đọc một ảnh và chuyển thành vector 1D
# ===========================================================================
def load_image_as_vector(image_path: str) -> np.ndarray:
    """
    Đọc một file ảnh, chuyển sang grayscale, và duỗi thành vector 1D.

    Đây là thao tác "vector hóa" (vectorization) — bước đầu tiên của thuật
    toán Eigenfaces: mỗi ảnh 2D được biểu diễn như một điểm trong không gian
    R^(IMG_SIZE).

    Tham số:
        image_path (str): đường dẫn tới file ảnh (.pgm, .jpg, .png, ...).

    Trả về:
        np.ndarray, shape (IMG_SIZE,), dtype float64
            Vector pixel, giá trị trong [0.0, 255.0].
    """
    # Bước A: Mở ảnh bằng thư viện Pillow
    img = Image.open(image_path)

    # Bước B: Chuyển sang grayscale
    # Mode 'L' = 8-bit luminance (mỗi pixel là 1 giá trị 0–255)
    # Nếu ảnh đã là grayscale (mode='L') thì convert() không làm gì thêm.
    img_gray = img.convert("L")

    # Bước C: Resize về kích thước chuẩn (IMG_WIDTH x IMG_HEIGHT)
    # Cần thiết để đảm bảo tất cả vector có cùng độ dài, nhất là khi dùng
    # ảnh tự chụp có kích thước khác với ảnh ORL gốc.
    img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT))

    # Bước D: Chuyển sang numpy array 2D, shape (IMG_HEIGHT, IMG_WIDTH)
    img_array = np.array(img_resized, dtype=np.float64)

    # Bước E: Duỗi (flatten) từ 2D → 1D, shape (IMG_SIZE,)
    # flatten() theo thứ tự hàng (row-major / C order):
    # [pixel(0,0), pixel(0,1), ..., pixel(111,91)]
    img_vector = img_array.flatten()

    return img_vector


# ===========================================================================
# HÀM 3: Tải toàn bộ ORL dataset và chia train/test
# ===========================================================================
def load_orl_dataset(
    data_dir: str = "data/orl_faces",
    n_persons: int = 40,
    n_images_per_person: int = 10,
    n_test_per_person: int = 2,
) -> tuple:
    """
    Đọc toàn bộ ORL Faces dataset, vector hóa từng ảnh, và chia train/test.

    Quy tắc chia:
        - n_test_per_person ảnh CUỐI của mỗi người → tập test
        - Phần còn lại → tập train
        Ví dụ mặc định: 2 ảnh cuối (ảnh 9, 10) → test; 8 ảnh đầu → train
        → X_train: (40×8, 10304) = (320, 10304)
        → X_test : (40×2, 10304) = (80, 10304)

    Tham số:
        data_dir (str)            : đường dẫn thư mục chứa s1..s40.
        n_persons (int)           : số người (mặc định 40).
        n_images_per_person (int) : số ảnh/người (mặc định 10).
        n_test_per_person (int)   : số ảnh/người dùng để test (mặc định 2).

    Trả về:
        X_train (np.ndarray): shape (N_train, IMG_SIZE) — ảnh huấn luyện.
        y_train (np.ndarray): shape (N_train,)          — nhãn person (1..40).
        X_test  (np.ndarray): shape (N_test,  IMG_SIZE) — ảnh kiểm tra.
        y_test  (np.ndarray): shape (N_test,)           — nhãn tập test.
    """
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    # Ngưỡng phân chia: ảnh có chỉ số > cutoff → test
    cutoff = n_images_per_person - n_test_per_person  # = 8

    for person_id in range(1, n_persons + 1):
        person_dir = os.path.join(data_dir, f"s{person_id}")

        if not os.path.isdir(person_dir):
            print(f"[WARN] Bỏ qua — không tìm thấy thư mục: {person_dir}")
            continue

        for img_idx in range(1, n_images_per_person + 1):
            img_path = os.path.join(person_dir, f"{img_idx}.pgm")

            if not os.path.exists(img_path):
                print(f"[WARN] Bỏ qua — không tìm thấy file: {img_path}")
                continue

            # Vector hóa ảnh
            vec = load_image_as_vector(img_path)

            # Gán vào train hoặc test
            if img_idx > cutoff:
                X_test.append(vec)
                y_test.append(person_id)
            else:
                X_train.append(vec)
                y_train.append(person_id)

    # Chuyển list of vectors thành ma trận numpy
    # Mỗi HÀNG là một ảnh (một điểm dữ liệu trong R^IMG_SIZE)
    X_train = np.array(X_train, dtype=np.float64)  # shape: (N_train, 10304)
    y_train = np.array(y_train, dtype=np.int32)     # shape: (N_train,)
    X_test  = np.array(X_test,  dtype=np.float64)   # shape: (N_test,  10304)
    y_test  = np.array(y_test,  dtype=np.int32)     # shape: (N_test,)

    return X_train, y_train, X_test, y_test


# ===========================================================================
# HÀM 4 (tiện ích): In thống kê dataset
# ===========================================================================
def print_dataset_info(X_train, y_train, X_test, y_test) -> None:
    """In thống kê tóm tắt của dataset đã tải."""
    n_train   = X_train.shape[0]
    n_test    = X_test.shape[0]
    n_classes = len(np.unique(y_train)) if len(y_train) > 0 else 0
    h, w      = IMG_HEIGHT, IMG_WIDTH

    print("\n" + "=" * 45)
    print("THỐNG KÊ DATASET ORL FACES")
    print("=" * 45)
    print(f"  Kích thước ảnh gốc : {h} x {w} pixel")
    print(f"  Kích thước vector  : {IMG_SIZE} chiều")
    print(f"  Số người (classes) : {n_classes}")
    print(f"  Tập huấn luyện     : {n_train} ảnh  — shape {X_train.shape}")
    print(f"  Tập kiểm tra       : {n_test}  ảnh  — shape {X_test.shape}")
    if n_train > 0:
        print(f"  Giá trị pixel      : [{X_train.min():.0f}, {X_train.max():.0f}]")
    else:
        print(f"  Giá trị pixel      : [—, —] (không có dữ liệu)")
    print("=" * 45 + "\n")


# ===========================================================================
# CHẠY THỬ (demo khi chạy file này trực tiếp)
# ===========================================================================
if __name__ == "__main__":
    import sys

    # Xác định thư mục gốc của dự án (một cấp trên thư mục src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, "data", "orl_faces")

    # Tải dataset nếu chưa có
    if not os.path.isdir(data_dir):
        download_orl_dataset(save_dir=os.path.join(project_root, "data"))

    # Tải và chia dữ liệu
    print("Đang đọc dataset...")
    X_train, y_train, X_test, y_test = load_orl_dataset(data_dir=data_dir)

    # In thống kê
    print_dataset_info(X_train, y_train, X_test, y_test)

    # Kiểm tra nhanh: in giá trị của ảnh đầu tiên trong tập train
    print(f"Ảnh đầu tiên (người {y_train[0]}): vector shape = {X_train[0].shape}")
    print(f"  5 giá trị pixel đầu tiên: {X_train[0][:5]}")
    print(f"  5 giá trị pixel cuối:     {X_train[0][-5:]}")
    print("\nBước 1 hoàn tất. Sẵn sàng cho Bước 2 (Eigenfaces Core).")
