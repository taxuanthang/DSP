import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import các hàm toán học lõi mà bạn đã đóng gói chuẩn ở các bước trước
from src.filters import (
    butterworth_lowpass_filter,
    gaussian_highpass_filter
)

from src.degradation import(    
    apply_motion_blur,
    wiener_deconvolution
)
class DSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HUST DSP - Hệ Thống Xử Lý Ảnh Miền Tần Số")
        self.root.geometry("1200x750")
        
        # Biến lưu trữ dữ liệu ảnh và ma trận bổ trợ
        self.img_origin = None      # Ảnh gốc (uint8)
        self.img_degraded_float = None  # Ảnh bị lỗi dạng float64 (cho Wiener)
        self.F_psf = None           # Phổ ma trận nhòe PSF
        
        self.create_widgets()

    def create_widgets(self):
        # ------------------ PANEL ĐIỀU KHIỂN (BÊN TRÁI) ------------------
        control_panel = tk.LabelFrame(self.root, text=" Chức năng & Tham số ", padx=10, pady=10)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Nút chọn ảnh đầu vào
        btn_load = tk.Button(control_panel, text="📁 Chọn ảnh đầu vào", command=self.load_image, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'))
        btn_load.pack(fill=tk.X, pady=5)

        # ---- NHÓM 1: LỌC TẦN SỐ CƠ BẢN ----
        group_freq = tk.LabelFrame(control_panel, text="1. Lọc Miền Tần Số", padx=5, pady=5)
        group_freq.pack(fill=tk.X, pady=10)

        tk.Label(group_freq, text="Bán kính cắt (D0):").pack(anchor=tk.W)
        self.slider_radius = tk.Scale(group_freq, from_=1, to=150, orient=tk.HORIZONTAL)
        self.slider_radius.set(40)
        self.slider_radius.pack(fill=tk.X, pady=2)

        tk.Label(group_freq, text="Bậc n (Chỉ dùng cho Butterworth):").pack(anchor=tk.W)
        self.slider_n = tk.Scale(group_freq, from_=1, to=10, orient=tk.HORIZONTAL)
        self.slider_n.set(2)
        self.slider_n.pack(fill=tk.X, pady=2)

        btn_bw = tk.Button(group_freq, text="Chạy Butterworth Lowpass", command=self.run_butterworth)
        btn_bw.pack(fill=tk.X, pady=3)

        btn_gauss = tk.Button(group_freq, text="Chạy Gaussian Highpass", command=self.run_gaussian_highpass)
        btn_gauss.pack(fill=tk.X, pady=3)


        # ---- NHÓM 2: PHỤC HỒI ẢNH WIENER ----
        group_wiener = tk.LabelFrame(control_panel, text="2. Phục Hồi Ảnh (Wiener)", padx=5, pady=5)
        group_wiener.pack(fill=tk.X, pady=10)

        tk.Label(group_wiener, text="Độ dài vệt nhòe (L):").pack(anchor=tk.W)
        self.slider_L = tk.Scale(group_wiener, from_=5, to=100, orient=tk.HORIZONTAL)
        self.slider_L.set(30)
        self.slider_L.pack(fill=tk.X, pady=2)

        tk.Label(group_wiener, text="Độ nhiễu (Noise Sigma):").pack(anchor=tk.W)
        self.slider_sigma = tk.Scale(group_wiener, from_=0.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)
        self.slider_sigma.set(2.0)
        self.slider_sigma.pack(fill=tk.X, pady=2)

        btn_blur = tk.Button(group_wiener, text="Bước 1: Tạo ảnh lỗi (Blur + Nhiễu)", command=self.run_make_blur, bg="#FF9800", fg="white")
        btn_blur.pack(fill=tk.X, pady=4)

        btn_wiener = tk.Button(group_wiener, text="Bước 2: Quét Wiener tối ưu SSIM", command=self.run_wiener, bg="#2196F3", fg="white")
        btn_wiener.pack(fill=tk.X, pady=4)

        # Trạng thái hệ thống
        self.lbl_status = tk.Label(control_panel, text="Trạng thái: Sẵn sàng", fg="blue", font=('Arial', 9, 'italic'))
        self.lbl_status.pack(anchor=tk.W, pady=10)


        # ------------------ KHU VỰC HIỂN THỊ ẢNH (BÊN PHẢI) ------------------
        display_panel = tk.Frame(self.root)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Khung ảnh 1: Ảnh gốc
        self.frame_img1 = tk.LabelFrame(display_panel, text=" Ảnh gốc / Ảnh suy biến ")
        self.frame_img1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_img1 = tk.Label(self.frame_img1)
        self.lbl_img1.pack()

        # Khung ảnh 2: Ảnh kết quả sau xử lý
        self.frame_img2 = tk.LabelFrame(display_panel, text=" Ảnh kết quả đầu ra ")
        self.frame_img2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.lbl_img2 = tk.Label(self.frame_img2)
        self.lbl_img2.pack()

        # Khung ảnh 3: Phổ tần số miền Fourier
        self.frame_img3 = tk.LabelFrame(display_panel, text=" Phổ biên độ tần số (Fourier Spectrum) ")
        self.frame_img3.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.lbl_img3 = tk.Label(self.frame_img3)
        self.lbl_img3.pack()

        # Cấu hình grid tỉ lệ đều nhau
        display_panel.rowconfigure(0, weight=1)
        display_panel.rowconfigure(1, weight=1)
        display_panel.columnconfigure(0, weight=1)
        display_panel.columnconfigure(1, weight=1)

    # ---- CÁC HÀM XỬ LÝ SỰ KIỆN ----
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            try:
                # SỬA Ở ĐÂY: Sử dụng np.fromfile và cv2.imdecode để đọc đường dẫn có dấu tiếng Việt
                # Đọc file thô dưới dạng mảng uint8
                img_array = np.fromfile(file_path, dtype=np.uint8)
                # Giải mã mảng thô thành ảnh xám (IMREAD_GRAYSCALE = 0)
                self.img_origin = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if self.img_origin is not None:
                    self.show_image(self.img_origin, self.lbl_img1)
                    self.lbl_status.config(text="Đã tải ảnh thành công (Hỗ trợ tiếng Việt)!", fg="green")
                    self.img_degraded_float = None
                    self.F_psf = None
                else:
                    messagebox.showerror("Lỗi", "Không thể giải mã file ảnh này!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")

    def show_image(self, cv_img, label_widget):
        # Chuyển đổi ma trận OpenCV sang đối tượng ảnh của Tkinter hiển thị
        h, w = cv_img.shape
        max_size = 280
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
            
        img_pil = Image.fromarray(cv_img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    def run_butterworth(self):
        if self.img_origin is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")
            return
        
        radius = self.slider_radius.get()
        n = self.slider_n.get()
        self.lbl_status.config(text="Đang chạy Butterworth Lowpass...", fg="blue")
        self.root.update()

        # Gọi hàm backend của bạn
        img_out, _, spec_filt, _ = butterworth_lowpass_filter(self.img_origin, radius=radius, n=n)
        
        self.show_image(img_out, self.lbl_img2)
        self.show_image(spec_filt, self.lbl_img3)
        self.lbl_status.config(text=f"Đã lọc Butterworth (D0={radius}, n={n})", fg="green")

    def run_gaussian_highpass(self):
        if self.img_origin is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")
            return
        
        radius = self.slider_radius.get()
        self.lbl_status.config(text="Đang chạy Gaussian Highpass...", fg="blue")
        self.root.update()

        # Gọi hàm backend của bạn
        img_out, _, spec_filt, _ = gaussian_highpass_filter(self.img_origin, radius=radius)
        
        self.show_image(img_out, self.lbl_img2)
        self.show_image(spec_filt, self.lbl_img3)
        self.lbl_status.config(text=f"Đã lọc Gaussian Highpass (D0={radius})", fg="green")

    def run_make_blur(self):
        if self.img_origin is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")
            return
        
        L = self.slider_L.get()
        sigma = self.slider_sigma.get()
        
        # Chạy thuật toán tạo lỗi
        img_deg_float, img_deg_uint8, F_psf = apply_motion_blur(self.img_origin, length=L, noise_sigma=sigma)
        
        # Lưu trữ ma trận float và phổ phục vụ cho bước Wiener liền mạch tiếp theo
        self.img_degraded_float = img_deg_float
        self.F_psf = F_psf
        
        # Hiển thị ảnh bị suy biến sang khung bên trái (thay thế ảnh gốc) để trực quan hóa
        self.show_image(img_deg_uint8, self.lbl_img1)
        self.lbl_status.config(text=f"Đã tạo lỗi: Nhòe L={L} + Nhiễu σ={sigma}", fg="red")

    def run_wiener(self):
        if self.img_origin is None or self.img_degraded_float is None or self.F_psf is None:
            messagebox.showwarning("Cảnh báo", "Bạn phải bấm tạo ảnh lỗi ở Bước 1 trước khi khôi phục!")
            return
        
        self.lbl_status.config(text="Đang quét tìm hằng số K tối ưu theo SSIM...", fg="blue")
        self.root.update()

        # Danh sách tham số K để thuật toán tự quét
        K_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
        
        # Gọi hàm giải mã Wiener tối ưu trực tiếp ma trận float của bạn
        best_img, opt_K, max_ssim = wiener_deconvolution(
            self.img_degraded_float, self.F_psf, self.img_origin, K_candidates=K_list
        )
        
        # Hiển thị ảnh khôi phục đẹp nhất lên khung kết quả
        self.show_image(best_img, self.lbl_img2)
        
        # Hiển thị phổ năng lượng của hàm giải mã lên khung phổ
        spec_filt = np.log(1 + np.abs(self.F_psf))
        spec_filt_uint8 = cv2.normalize(spec_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.show_image(spec_filt_uint8, self.lbl_img3)
        
        self.lbl_status.config(text=f"Khôi phục xong! K tối ưu = {opt_K} | SSIM đạt = {max_ssim:.4f}", fg="green")

if __name__ == "__main__":
    # Thiết lập chạy từ thư mục gốc của dự án để nhận diện đường dẫn src
    root = tk.Tk()
    app = DSPApp(root)
    root.mainloop()