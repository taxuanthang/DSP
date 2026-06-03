import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import traceback

# Kiểm tra import các file xử lý thuật toán cốt lõi
is_demo = False
try:
    from src.filters import (
        ideal_lowpass_filter,
        butterworth_lowpass_filter,
        ideal_highpass_filter,
        gaussian_highpass_filter,
        notch_reject_filter
    )
    from src.degradation import (
        apply_motion_blur,
        apply_defocus_blur,
        wiener_deconvolution
    )
except ImportError as e:
    is_demo = True
    print(f"Lỗi import thuật toán, chuyển sang chế độ demo: {e}")
    # Định nghĩa thô sơ để tránh crash giao diện khi thiếu file src
    def ideal_lowpass_filter(img, radius): return cv2.blur(img, (5, 5)), None, img, None
    def butterworth_lowpass_filter(img, radius, n): return cv2.blur(img, (9, 9)), None, img, None
    def ideal_highpass_filter(img, radius): return cv2.Canny(img, 50, 150), None, img, None
    def gaussian_highpass_filter(img, radius): return cv2.Canny(img, 30, 100), None, img, None
    def notch_reject_filter(img, uk, vk, radius): return cv2.equalizeHist(img), None, img, None
    def apply_motion_blur(img, length, noise_sigma): return img.astype(np.float32), img, np.ones(img.shape)
    def apply_defocus_blur(img, radius, noise_sigma): return img.astype(np.float32), img, np.ones(img.shape)
    def wiener_deconvolution(img_deg, F_psf, img_orig, K_candidates): return img_orig, 0.01, 1.0

try:
    from src.opticalFlowEstimation import horn_schunck_optical_flow
except ImportError:
    def horn_schunck_optical_flow(img1, img2, alpha=1.0, num_iters=100):
        return np.zeros(img1.shape), np.zeros(img1.shape)


class DSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HUST DSP - Hệ Thống Xử Lý Ảnh Miền Tần Số & Phân Tích Chuyển Động")
        self.root.geometry("1350x820")
        
        # Quản lý trạng thái dữ liệu ảnh và video
        self.img_origin = None          
        self.img_filtered = None        
        self.img_blur_raw = None  
        self.F_psf = None               
        self.notch_coords = (30, 30)
        
        self.is_playing_video = False
        self.cap = None
        self.prev_gray = None
        self.motion_history = []
        self.frame_count = 0

        self.preview_job = None

        self.blur_type = tk.StringVar(value="Motion Blur")
        self.setup_ui()
        
        if is_demo:
            messagebox.showwarning("Cảnh báo", "Không kết nối được các hàm xử lý của src. Hệ thống chạy ở chế độ DEMO.")

    def setup_ui(self):
        # Thanh điều khiển chức năng bên trái
        sidebar = tk.LabelFrame(self.root, text=" Chức năng & Tham số ", padx=10, pady=5)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Cụm nút Thao tác Tệp
        f_io = tk.LabelFrame(sidebar, text="Thao tác Tệp", padx=5, pady=3)
        f_io.pack(fill=tk.X, pady=3)
        tk.Button(f_io, text="📁 Chọn ảnh đầu vào", command=self.load_image, bg="#4CAF50", fg="white", font=('Arial', 9, 'bold')).pack(fill=tk.X, pady=2)
        tk.Button(f_io, text="💾 Xuất ảnh kết quả", command=self.export_result, bg="#9C27B0", fg="white", font=('Arial', 9, 'bold')).pack(fill=tk.X, pady=2)

        # Khối chức năng 1: Lọc miền tần số
        f_freq = tk.LabelFrame(sidebar, text="1. Lọc Miền Tần Số", padx=5, pady=3)
        f_freq.pack(fill=tk.X, pady=3)

        tk.Label(f_freq, text="Chọn loại bộ lọc:").pack(anchor=tk.W)
        self.filter_types = [
            "Ideal Lowpass (Thông thấp lý tưởng)",
            "Butterworth Lowpass (Thông thấp Butterworth)",
            "Ideal Highpass (Thông cao lý tưởng)",
            "Gaussian Highpass (Thông cao Gaussian)",
            "2D Notch Filter (Bộ lọc chắn nhiễu dải)"
        ]
        self.selected_filter = tk.StringVar(value=self.filter_types[0])
        
        cb_filter = tk.OptionMenu(f_freq, self.selected_filter, *self.filter_types, command=self.on_filter_change)
        cb_filter.config(width=28, anchor=tk.W)
        cb_filter.pack(fill=tk.X, pady=2)

        tk.Label(f_freq, text="Bán kính cắt (D0):").pack(anchor=tk.W)
        self.slider_radius = tk.Scale(
            f_freq,
            from_=1,
            to=150,
            orient=tk.HORIZONTAL,
            command=self.schedule_filter_preview
        )
        self.slider_radius.set(40)
        self.slider_radius.pack(fill=tk.X)

        self.frame_param_n = tk.Frame(f_freq)
        tk.Label(self.frame_param_n, text="Bậc n (Chỉ cho Butterworth):").pack(anchor=tk.W)
        self.slider_n = tk.Scale(
            self.frame_param_n,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            command=self.schedule_filter_preview
        )
        self.slider_n.set(2)
        self.slider_n.pack(fill=tk.X)

        self.lbl_notch_hint = tk.Label(f_freq, text="💡 Notch: Click chuột vào ô số 2\ntrên màn hình để đổi tâm nhiễu.", 
                                       fg="#FF5722", font=('Arial', 8, 'italic'), justify=tk.LEFT)

        self.btn_run_filter = tk.Button(f_freq, text="⚡ Thực thi bộ lọc tần số", command=self.run_frequency_filter, bg="#2196F3", fg="white", font=('Arial', 9, 'bold'))
        self.btn_run_filter.pack(fill=tk.X, pady=5)

        # Khối chức năng 2: Phục hồi Wiener
        f_wiener = tk.LabelFrame(sidebar, text="2. Phục Hồi Ảnh (Wiener)", padx=5, pady=3)
        f_wiener.pack(fill=tk.X, pady=3)

        tk.Label(f_wiener, text="Loại hiệu ứng nhòe:").pack(anchor=tk.W)
        tk.OptionMenu(f_wiener, self.blur_type, "Motion Blur", "Defocus Blur").pack(fill=tk.X)

        tk.Label(f_wiener, text="Độ dài vệt nhòe (L):").pack(anchor=tk.W)
        self.slider_L = tk.Scale(f_wiener, from_=5, to=100, orient=tk.HORIZONTAL)
        self.slider_L.set(30)
        self.slider_L.pack(fill=tk.X)

        tk.Label(f_wiener, text="Bán kính phá nét (r):").pack(anchor=tk.W)
        self.slider_defocus = tk.Scale(f_wiener, from_=1, to=50, orient=tk.HORIZONTAL)
        self.slider_defocus.set(15)
        self.slider_defocus.pack(fill=tk.X)

        tk.Label(f_wiener, text="Cường độ nhiễu (Sigma):").pack(anchor=tk.W)
        self.slider_sigma = tk.Scale(f_wiener, from_=0.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)
        self.slider_sigma.set(2.0)
        self.slider_sigma.pack(fill=tk.X)

        tk.Button(f_wiener, text="Bước 1: Tạo nhiễu + nhòe", command=self.run_make_blur, bg="#FF9800", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(f_wiener, text="Bước 2: Phục hồi Wiener", command=self.run_wiener, bg="#00BCD4", fg="white").pack(fill=tk.X, pady=2)

        # Khối chức năng 3: Phân tích dòng quang học Optical Flow
        f_of = tk.LabelFrame(sidebar, text="3. Phân Tích Chuyển Động (Optical Flow)", padx=5, pady=3)
        f_of.pack(fill=tk.X, pady=3)

        tk.Label(f_of, text="Hệ số mượt (Alpha):").pack(anchor=tk.W)
        self.slider_alpha = tk.Scale(f_of, from_=1.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)
        self.slider_alpha.set(7.0)
        self.slider_alpha.pack(fill=tk.X)

        tk.Label(f_of, text="Vòng lặp (Iters):").pack(anchor=tk.W)
        self.slider_iters = tk.Scale(f_of, from_=10, to=150, resolution=10, orient=tk.HORIZONTAL)
        self.slider_iters.set(30)
        self.slider_iters.pack(fill=tk.X)

        tk.Label(f_of, text="Ngưỡng vector hiển thị:").pack(anchor=tk.W)
        self.slider_thresh = tk.Scale(f_of, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.slider_thresh.set(2.5)
        self.slider_thresh.pack(fill=tk.X)

        tk.Button(f_of, text="📂 Chọn tập tin Video", command=self.load_video_file, bg="#009688", fg="white").pack(fill=tk.X, pady=1)
        self.btn_video = tk.Button(f_of, text="🎥 Phân tích dòng Camera", command=self.toggle_video_flow, bg="#E91E63", fg="white", font=('Arial', 9, 'bold'))
        self.btn_video.pack(fill=tk.X, pady=1)
        tk.Button(f_of, text="📊 Xem đồ thị chuyển động", command=self.plot_motion_curve, bg="#607D8B", fg="white").pack(fill=tk.X, pady=1)

        # Label hiển thị trạng thái hệ thống
        self.lbl_status = tk.Label(sidebar, text="Trạng thái: Sẵn sàng", fg="blue", font=('Arial', 9, 'italic'), wraplength=230, justify=tk.LEFT)
        self.lbl_status.pack(anchor=tk.W, pady=5)

        # Lưới hiển thị kết quả trực quan (2x2) ở bên phải
        grid_panel = tk.Frame(self.root)
        grid_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frames_data = [
            (" 1. Khung ảnh gốc / Hiện tại (Original) ", 0, 0),
            (" 2. Bản đồ phổ tần số (Spectrum) ", 0, 1),
            (" 3. Phổ sau lọc / Bản đồ nhiệt (Magnitude Map) ", 1, 0),
            (" 4. Kết quả / Trường Vector (Filtered / Quiver) ", 1, 1)
        ]
        
        self.labels_view = []
        for text, r, c in frames_data:
            frame = tk.LabelFrame(grid_panel, text=text)
            frame.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            lbl = tk.Label(frame)
            lbl.pack(expand=True)
            self.labels_view.append(lbl)
            
        self.lbl_img1, self.lbl_img2, self.lbl_img3, self.lbl_img4 = self.labels_view
        self.lbl_img2.bind("<Button-1>", self.on_spectrum_click)

        for i in range(2):
            grid_panel.rowconfigure(i, weight=1)
            grid_panel.columnconfigure(i, weight=1)

    def schedule_filter_preview(self, event=None):

        if self.img_origin is None:
            return

        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)

        self.preview_job = self.root.after(
            10,
            self.run_frequency_filter
        )

    def on_filter_change(self, value):
        self.frame_param_n.pack_forget()
        self.lbl_notch_hint.pack_forget()
        self.btn_run_filter.pack_forget()

        if "Butterworth" in value:
            self.frame_param_n.pack(fill=tk.X, pady=2)
        elif "Notch" in value:
            self.lbl_notch_hint.pack(fill=tk.X, pady=2)
            
        self.btn_run_filter.pack(fill=tk.X, pady=5)

        if self.img_origin is not None:
            self.schedule_filter_preview()

    def on_spectrum_click(self, event):
        if self.img_origin is None or "Notch" not in self.selected_filter.get():
            return
        lbl_w, lbl_h = self.lbl_img2.winfo_width(), self.lbl_img2.winfo_height()
        img_h, img_w = self.img_origin.shape[:2]
        scale = 280 / max(img_h, img_w)
        rw, rh = int(img_w * scale), int(img_h * scale)
        sx, sy = (lbl_w - rw) // 2, (lbl_h - rh) // 2
        cx, cy = event.x - sx, event.y - sy
        
        if 0 <= cx < rw and 0 <= cy < rh:
            self.notch_coords = (int(cx / scale) - (img_w // 2), int(cy / scale) - (img_h // 2))
            self.lbl_status.config(text=f"📍 Tọa độ Notch: uk={self.notch_coords[0]}, vk={self.notch_coords[1]}", fg="#FF5722")
            self.run_frequency_filter()

    def run_frequency_filter(self):
        self.root.config(cursor="watch")
        if self.img_origin is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")
            return
            
        mode = self.selected_filter.get()
        radius = self.slider_radius.get()
        self.lbl_status.config(text="Đang tính toán bộ lọc...", fg="blue")
        self.root.update()

        try:
            if "Ideal Lowpass" in mode:
                img_out, _, spec_filt, _ = ideal_lowpass_filter(self.img_origin, radius=radius)
            elif "Butterworth Lowpass" in mode:
                img_out, _, spec_filt, _ = butterworth_lowpass_filter(self.img_origin, radius=radius, n=self.slider_n.get())
            elif "Ideal Highpass" in mode:
                img_out, _, spec_filt, _ = ideal_highpass_filter(self.img_origin, radius=radius)
            elif "Gaussian Highpass" in mode:
                img_out, _, spec_filt, _ = gaussian_highpass_filter(self.img_origin, radius=radius)
            elif "2D Notch Filter" in mode:
                uk, vk = self.notch_coords
                img_out, _, spec_filt, _ = notch_reject_filter(self.img_origin, uk=uk, vk=vk, radius=radius)
            
            if img_out is None:
                raise ValueError("Thuật toán trả về dữ liệu rỗng (None)!")

            self.img_filtered = img_out

            if spec_filt is not None:
                spec_abs = np.abs(spec_filt)
                if np.max(spec_abs) > 255 or np.max(spec_abs) < 10:
                    spec_abs = np.log(1 + spec_abs)
                spec_out = cv2.normalize(spec_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                F_c = np.fft.fftshift(np.fft.fft2(img_out.astype(np.float64)))
                spec_out = cv2.normalize(np.log(1 + np.abs(F_c)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            self.show_image(spec_out, self.lbl_img3)
            self.show_image(cv2.normalize(np.abs(img_out), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.lbl_img4)
            self.lbl_status.config(text=f"Đã áp dụng bộ lọc {mode.split(' (')[0]}", fg="green")

        except Exception as err:
            messagebox.showerror("Lỗi thuật toán", f"Lỗi xảy ra trong quá trình tính toán:\n{err}")
            self.lbl_status.config(text="Tính toán thất bại!", fg="red")

        self.root.config(cursor="")

    def run_make_blur(self):
        if self.img_origin is None:
            messagebox.showwarning("Cảnh báo", "Hãy tải ảnh đầu vào lên trước!")
            return

        sigma = self.slider_sigma.get()
        mode = self.blur_type.get()
        self.lbl_status.config(text="Đang làm nhòe ảnh mô phỏng...", fg="blue")
        self.root.update()

        if mode == "Motion Blur":
            L = self.slider_L.get()
            raw_f, out_u8, psf_f = apply_motion_blur(self.img_origin, length=L, noise_sigma=sigma)
            msg = f"Motion Blur L={L}"
        else:
            r = self.slider_defocus.get()
            raw_f, out_u8, psf_f = apply_defocus_blur(self.img_origin, radius=r, noise_sigma=sigma)
            msg = f"Defocus Blur r={r}"

        self.img_blur_raw = raw_f
        self.F_psf = psf_f
        self.show_image(out_u8, self.lbl_img1)
        self.lbl_status.config(text=f"Đã tạo: {msg}", fg="brown")

    def run_wiener(self):
        if self.img_origin is None or self.img_blur_raw is None: 
            messagebox.showwarning("Cảnh báo", "Vui lòng chạy bước tạo ảnh lỗi trước!")
            return
        self.lbl_status.config(text="Đang tối ưu hóa tham số Wiener...", fg="blue")
        self.root.update()
        
        best_img, opt_K, max_ssim, best_psnr, best_snr = wiener_deconvolution(self.img_blur_raw, self.F_psf, self.img_origin, K_candidates=[0.0001, 0.005, 0.01, 0.05, 0.1])
        self.img_filtered = best_img
        self.show_image(cv2.normalize(np.abs(best_img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.lbl_img4)
        self.lbl_status.config(
            text=
            f"Wiener OK | "
            f"K={opt_K} | "
            f"SSIM={max_ssim:.4f} | "
            f"PSNR={best_psnr:.2f}dB | "
            f"SNR={best_snr:.2f}dB",
            fg="green"
        )

    def load_video_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
        if path:
            if self.is_playing_video:
                self.toggle_video_flow()
            self.video_source = path
            self.motion_history = [] 
            self.frame_count = 0
            self.lbl_status.config(text=f"Đã nạp video: {os.path.basename(path)}", fg="green")
            self.btn_video.config(text="▶ Khởi chạy Phân Tích Video", bg="#2196F3")
            
    def toggle_video_flow(self):
        if not self.is_playing_video:
            src = getattr(self, 'video_source', 0)
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không mở được camera hoặc nguồn dữ liệu video!")
                return
            
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                return
                
            self.prev_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
            self.is_playing_video = True
            self.motion_history = [] 
            self.frame_count = 0
            self.btn_video.config(text="⏹ Dừng phân tích", bg="#f44336")
            self.update_video_frame()
        else:
            self.is_playing_video = False
            if self.cap:
                self.cap.release()
            self.video_source = 0 
            self.btn_video.config(text="🎥 Phân tích dòng Camera", bg="#E91E63")

    def update_video_frame(self):
        if not self.is_playing_video:
            return
            
        ret, frame = self.cap.read()
        if not ret or self.frame_count >= 30:
            self.is_playing_video = False
            if self.cap:
                self.cap.release()
            self.btn_video.config(text="🎥 Phân tích dòng Camera", bg="#E91E63")
            
            if self.motion_history:
                mx_idx = int(np.argmax(self.motion_history))
                self.lbl_status.config(text=f"✅ Xong 30 frames! Khung cực đại: Frame {mx_idx + 1} ({self.motion_history[mx_idx]:.3f})", fg="green")
                messagebox.showinfo("Hoàn tất", f"Đã quét đủ 30 khung hình!\nChuyển động đỉnh lớn nhất tại Frame: {mx_idx + 1}")
            return
            
        self.frame_count += 1
        curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
        
        p_blur = cv2.GaussianBlur(self.prev_gray, (5, 5), 0)
        c_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        u, v = horn_schunck_optical_flow(p_blur, c_blur, alpha=self.slider_alpha.get(), num_iters=self.slider_iters.get())
        
        mag = np.sqrt(u**2 + v**2)
        mean_v = float(np.mean(mag))
        self.motion_history.append(mean_v)
        
        mag_col = cv2.applyColorMap(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        quiver = self.draw_filtered_quiver(curr_gray, u, v, step=16, threshold=self.slider_thresh.get())
        
        spec_curr = cv2.normalize(np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(curr_gray.astype(np.float64))))), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        self.show_image(curr_gray, self.lbl_img1)        
        self.show_image(spec_curr, self.lbl_img2)  
        self.show_image(mag_col, self.lbl_img3)        
        self.show_image(quiver, self.lbl_img4)     
        
        self.prev_gray = curr_gray
        self.lbl_status.config(text=f"Xử lý dữ liệu chuỗi: {self.frame_count}/30 | Biên độ: {mean_v:.3f}", fg="purple")
        self.root.after(15, self.update_video_frame)

    def draw_filtered_quiver(self, img, u, v, step, threshold):
        h, w = img.shape
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx, dy = u[y, x], v[y, x]
                if np.sqrt(dx**2 + dy**2) > threshold:
                    cv2.arrowedLine(out, (x, y), (int(x + dx * 3), int(y + dy * 3)), (0, 255, 0), 1, tipLength=0.3)
        return out

    def plot_motion_curve(self):
        if not self.motion_history: 
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu chuyển động để xuất biểu đồ.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError: 
            messagebox.showerror("Thất bại", "Môi trường thiếu thư viện matplotlib.")
            return
            
        y_data = np.array(self.motion_history)
        mx_idx = int(np.argmax(y_data))
        
        plt.figure(figsize=(9, 4.5))
        plt.plot(np.arange(1, len(self.motion_history) + 1), y_data, marker='o', color='b', linewidth=2)
        plt.scatter(mx_idx + 1, y_data[mx_idx], color='red', s=120, zorder=5, label=f'Đỉnh (Frame {mx_idx + 1})')
        plt.title('Video Motion Magnitude Curve (30 Frames)')
        plt.xlabel('Frame Number')
        plt.ylabel('Mean Motion Magnitude')
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.show()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            try:
                self.img_origin = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if self.img_origin is not None:
                    self.show_image(self.img_origin, self.lbl_img1)
                    F_orig = np.fft.fftshift(np.fft.fft2(self.img_origin.astype(np.float64)))
                    self.show_image(cv2.normalize(np.log(1 + np.abs(F_orig)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.lbl_img2)
                    
                    for lbl in [self.lbl_img3, self.lbl_img4]:
                        lbl.config(image='')
                        lbl.image = None
                    self.lbl_status.config(text="Đã tải ảnh gốc thành công!", fg="green")
                else:
                    messagebox.showerror("Lỗi", "Không nhận dạng được cấu trúc tệp ảnh!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không mở được tệp: {str(e)}")

    def export_result(self):
        if self.img_filtered is None:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu ảnh kết quả để kết xuất!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if path:
            ret, buf = cv2.imencode(".png", self.img_filtered)
            if ret:
                buf.tofile(path)
                messagebox.showinfo("Thông báo", "Lưu ảnh thành công!")

    def show_image(self, cv_img, label_widget):
        if cv_img is None:
            return
        h, w = cv_img.shape[:2]
        scale = 280 / max(h, w)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(cv_img, (int(w * scale), int(h * scale)))))
        label_widget.config(image=img_tk)
        label_widget.image = img_tk  
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = DSPApp(root)
    root.mainloop()