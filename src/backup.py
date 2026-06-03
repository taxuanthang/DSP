import os

import cv2

import numpy as np

import tkinter as tk

from tkinter import filedialog, messagebox

from PIL import Image, ImageTk



# Giữ nguyên cấu trúc import từ các file toán học của bạn

from src.filters import (

    butterworth_lowpass_filter,

    gaussian_highpass_filter

)



from src.degradation import (    

    apply_motion_blur,

    wiener_deconvolution

)



# Khối xử lý Horn-Schunck Optical Flow tích hợp chống lag và chỉ sai hướng

try:

    from src.optical_flow import horn_schunck_optical_flow

except ImportError:

    def horn_schunck_optical_flow(img1, img2, alpha=1.0, num_iters=100):

        I1 = img1.astype(np.float64)

        I2 = img2.astype(np.float64)

        kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25

        kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25

        kernel_t = np.array([[1, 1], [1, 1]]) * 0.25

        Ix = cv2.filter2D(I1, -1, kernel_x) + cv2.filter2D(I2, -1, kernel_x)

        Iy = cv2.filter2D(I1, -1, kernel_y) + cv2.filter2D(I2, -1, kernel_y)

        It = cv2.filter2D(I2, -1, kernel_t) - cv2.filter2D(I1, -1, kernel_t)

        u = np.zeros(I1.shape)

        v = np.zeros(I1.shape)

        kernel_avg = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

        denominator = alpha**2 + Ix**2 + Iy**2

        for _ in range(num_iters):

            u_avg = cv2.filter2D(u, -1, kernel_avg)

            v_avg = cv2.filter2D(v, -1, kernel_avg)

            brightness_error = (Ix * u_avg + Iy * v_avg + It) / denominator

            u = u_avg - Ix * brightness_error

            v = v_avg - Iy * brightness_error

        return u, v



class DSPApp:

    def __init__(self, root):

        self.root = root

        self.root.title("HUST DSP - Hệ Thống Xử Lý Ảnh Miền Tần Số & Phân Tích Chuyển Động")

        self.root.geometry("1280x780")

       

        # Biến lưu trữ dữ liệu

        self.img_origin = None          

        self.img_filtered = None        # Lưu ảnh đầu ra để phục vụ tính năng Export

        self.img_degraded_float = None  

        self.F_psf = None              

       

        # Luồng video

        self.is_playing_video = False

        self.cap = None

        self.prev_gray = None

       

        self.create_widgets()



    def create_widgets(self):

        # ================== PANEL ĐIỀU KHIỂN (BÊN TRÁI) ==================

        control_panel = tk.LabelFrame(self.root, text=" Chức năng & Tham số ", padx=10, pady=5)

        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)



        # Khối thao tác tệp

        group_file = tk.LabelFrame(control_panel, text="Thao tác Tệp", padx=5, pady=3)

        group_file.pack(fill=tk.X, pady=3)

       

        btn_load = tk.Button(group_file, text="📁 Chọn ảnh đầu vào", command=self.load_image, bg="#4CAF50", fg="white", font=('Arial', 9, 'bold'))

        btn_load.pack(fill=tk.X, pady=2)

       

        # ĐÁP ỨNG YÊU CẦU: Export result (Xuất ảnh kết quả)

        btn_export = tk.Button(group_file, text="💾 Xuất ảnh kết quả (Export)", command=self.export_result, bg="#9C27B0", fg="white", font=('Arial', 9, 'bold'))

        btn_export.pack(fill=tk.X, pady=2)



        # ---- NHÓM 1: LỌC TẦN SỐ CƠ BẢN ----

        group_freq = tk.LabelFrame(control_panel, text="1. Lọc Miền Tần Số", padx=5, pady=3)

        group_freq.pack(fill=tk.X, pady=3)



        tk.Label(group_freq, text="Bán kính cắt (D0):").pack(anchor=tk.W)

        self.slider_radius = tk.Scale(group_freq, from_=1, to=150, orient=tk.HORIZONTAL)

        self.slider_radius.set(40)

        self.slider_radius.pack(fill=tk.X)



        tk.Label(group_freq, text="Bậc n (Chỉ dùng cho Butterworth):").pack(anchor=tk.W)

        self.slider_n = tk.Scale(group_freq, from_=1, to=10, orient=tk.HORIZONTAL)

        self.slider_n.set(2)

        self.slider_n.pack(fill=tk.X)



        btn_bw = tk.Button(group_freq, text="Chạy Butterworth Lowpass", command=self.run_butterworth)

        btn_bw.pack(fill=tk.X, pady=2)



        btn_gauss = tk.Button(group_freq, text="Chạy Gaussian Highpass", command=self.run_gaussian_highpass)

        btn_gauss.pack(fill=tk.X, pady=2)



        # ---- NHÓM 2: PHỤC HỒI ẢNH WIENER ----

        group_wiener = tk.LabelFrame(control_panel, text="2. Phục Hồi Ảnh (Wiener)", padx=5, pady=3)

        group_wiener.pack(fill=tk.X, pady=3)



        tk.Label(group_wiener, text="Độ dài vệt nhòe (L):").pack(anchor=tk.W)

        self.slider_L = tk.Scale(group_wiener, from_=5, to=100, orient=tk.HORIZONTAL)

        self.slider_L.set(30)

        self.slider_L.pack(fill=tk.X)



        tk.Label(group_wiener, text="Độ nhiễu (Noise Sigma):").pack(anchor=tk.W)

        self.slider_sigma = tk.Scale(group_wiener, from_=0.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)

        self.slider_sigma.set(2.0)

        self.slider_sigma.pack(fill=tk.X)



        btn_blur = tk.Button(group_wiener, text="Bước 1: Tạo ảnh lỗi (Blur + Nhiễu)", command=self.run_make_blur, bg="#FF9800", fg="white")

        btn_blur.pack(fill=tk.X, pady=2)



        btn_wiener = tk.Button(group_wiener, text="Bước 2: Quét Wiener tối ưu SSIM", command=self.run_wiener, bg="#2196F3", fg="white")

        btn_wiener.pack(fill=tk.X, pady=2)



        # ---- NHÓM 3: OPTICAL FLOW ----

        group_of = tk.LabelFrame(control_panel, text="3. Video Optical Flow", padx=5, pady=3)

        group_of.pack(fill=tk.X, pady=3)



        tk.Label(group_of, text="Hệ số mượt mà (Alpha):").pack(anchor=tk.W)

        self.slider_alpha = tk.Scale(group_of, from_=1.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)

        self.slider_alpha.set(7.0)

        self.slider_alpha.pack(fill=tk.X)



        tk.Label(group_of, text="Số bước lặp (Iters):").pack(anchor=tk.W)

        # ĐÃ SỬA: Thay thế `step=10` bằng `resolution=10` để tránh TclError

        self.slider_of_iters = tk.Scale(group_of, from_=10, to=150, resolution=10, orient=tk.HORIZONTAL)

        self.slider_of_iters.set(30) # Để mức 30 mặc định để chạy mượt giảm lag

        self.slider_of_iters.pack(fill=tk.X)



        tk.Label(group_of, text="Ngưỡng lọc vector (Threshold):").pack(anchor=tk.W)

        self.slider_thresh = tk.Scale(group_of, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL)

        self.slider_thresh.set(2.5)

        self.slider_thresh.pack(fill=tk.X)



        tk.Label(group_of, text="Mật độ lưới vẽ (Step):").pack(anchor=tk.W)

        # ĐÃ SỬA: Thay thế `step=2` bằng `resolution=2` để tránh TclError

        self.slider_step = tk.Scale(group_of, from_=8, to=24, resolution=2, orient=tk.HORIZONTAL)

        self.slider_step.set(16)

        self.slider_step.pack(fill=tk.X)



        btn_select_video = tk.Button(group_of, text="📂 Chọn File Video (.mp4)", command=self.load_video_file, bg="#009688", fg="white")

        btn_select_video.pack(fill=tk.X, pady=1)



        self.btn_video = tk.Button(group_of, text="🎥 Bật Camera / Phân tích", command=self.toggle_video_flow, bg="#E91E63", fg="white", font=('Arial', 9, 'bold'))

        self.btn_video.pack(fill=tk.X, pady=1)



        # Trạng thái hệ thống

        self.lbl_status = tk.Label(control_panel, text="Trạng thái: Sẵn sàng", fg="blue", font=('Arial', 9, 'italic'))

        self.lbl_status.pack(anchor=tk.W, pady=5)





        # ================== KHU VỰC HIỂN THỊ LƯỚI 2×2 (BÊN PHẢI) ==================

        display_panel = tk.Frame(self.root)

        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)



        # Ô (0,0): Ảnh gốc (Original Image)

        self.frame_img1 = tk.LabelFrame(display_panel, text=" 1. Ảnh gốc / Khung hình đầu vào (Original) ")

        self.frame_img1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.lbl_img1 = tk.Label(self.frame_img1)

        self.lbl_img1.pack(expand=True)



        # Ô (0,1): Phổ ảnh gốc (Spectrum)

        self.frame_img2 = tk.LabelFrame(display_panel, text=" 2. Phổ biên độ tần số gốc (Spectrum) ")

        self.frame_img2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.lbl_img2 = tk.Label(self.frame_img2)

        self.lbl_img2.pack(expand=True)



        # Ô (1,0): Phổ sau khi lọc (Filtered Spectrum)

        self.frame_img3 = tk.LabelFrame(display_panel, text=" 3. Phổ biên độ sau lọc (Filtered Spectrum) ")

        self.frame_img3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.lbl_img3 = tk.Label(self.frame_img3)

        self.lbl_img3.pack(expand=True)



        # Ô (1,1): Ảnh kết quả sau lọc (Filtered Image)

        self.frame_img4 = tk.LabelFrame(display_panel, text=" 4. Ảnh kết quả đầu ra / Sơ đồ Vector (Filtered) ")

        self.frame_img4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.lbl_img4 = tk.Label(self.frame_img4)

        self.lbl_img4.pack(expand=True)



        # Định cấu hình chia đều tỉ lệ ma trận lưới 2x2

        display_panel.rowconfigure(0, weight=1)

        display_panel.rowconfigure(1, weight=1)

        display_panel.columnconfigure(0, weight=1)

        display_panel.columnconfigure(1, weight=1)





    # ---- CÁC HÀM XỬ LÝ ĐỒ HỌA VÀ ĐỌC FILE ----

    def load_image(self):

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

        if file_path:

            try:

                if self.is_playing_video:

                    self.toggle_video_flow()



                img_array = np.fromfile(file_path, dtype=np.uint8)

                self.img_origin = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

               

                if self.img_origin is not None:

                    # Hiển thị ảnh gốc vào Ô (0,0)

                    self.show_image(self.img_origin, self.lbl_img1)

                   

                    # Tính toán hiển thị Phổ gốc vào Ô (0,1)

                    F_orig = np.fft.fftshift(np.fft.fft2(self.img_origin.astype(np.float64)))

                    spec_orig = np.log(1 + np.abs(F_orig))

                    spec_orig_uint8 = cv2.normalize(spec_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    self.show_image(spec_orig_uint8, self.lbl_img2)

                   

                    # Xóa sạch dữ liệu cũ ở hàng dưới khi tải ảnh mới

                    self.lbl_img3.config(image='')

                    self.lbl_img4.config(image='')

                   

                    self.lbl_status.config(text="Đã tải ảnh gốc và tính toán phổ biên độ thành công!", fg="green")

                    self.img_degraded_float = None

                    self.F_psf = None

                    self.img_filtered = None

                else:

                    messagebox.showerror("Lỗi", "Không thể giải mã file ảnh!")

            except Exception as e:

                messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")



    def export_result(self):

        """Hàm xuất ảnh đầu ra lưu trữ xuống máy tính (Export)"""

        if self.img_filtered is None:

            messagebox.showwarning("Cảnh báo", "Không có dữ liệu ảnh kết quả để xuất! Vui lòng thực hiện lọc trước.")

            return

       

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])

        if file_path:

            try:

                # Lưu ảnh an toàn xử lý được cả đường dẫn tiếng Việt

                is_success, im_buf_arr = cv2.imencode(".png", self.img_filtered)

                if is_success:

                    im_buf_arr.tofile(file_path)

                    self.lbl_status.config(text=f"Đã xuất và lưu tệp kết quả tại: {os.path.basename(file_path)}", fg="green")

                    messagebox.showinfo("Thành công", "Đã lưu ảnh kết quả thành công!")

            except Exception as e:

                messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")



    def show_image(self, cv_img, label_widget):

        if len(cv_img.shape) == 3:

            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

           

        h, w = cv_img.shape[:2]

        max_size = 260 # Thu nhỏ nhẹ để vừa vặn hoàn hảo trong lưới bố cục 2x2

        if h > max_size or w > max_size:

            scale = max_size / max(h, w)

            cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))

           

        img_pil = Image.fromarray(cv_img)

        img_tk = ImageTk.PhotoImage(image=img_pil)

        label_widget.config(image=img_tk)

        label_widget.image = img_tk





    # ---- ĐIỀU KHIỂN CHẠY CÁC BỘ LỌC TẦN SỐ VÀ ĐẨY VÀO LƯỚI 2x2 ----

    def run_butterworth(self):

        if self.img_origin is None:

            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")

            return

        radius = self.slider_radius.get()

        n = self.slider_n.get()

        self.lbl_status.config(text="Đang chạy Butterworth Lowpass...", fg="blue")

        self.root.update()



        img_out, _, spec_filt, _ = butterworth_lowpass_filter(self.img_origin, radius=radius, n=n)

        self.img_filtered = img_out

       

        # Đẩy phổ sau lọc vào Ô (1,0) và Ảnh kết quả vào Ô (1,1)

        self.show_image(spec_filt, self.lbl_img3)

        self.show_image(img_out, self.lbl_img4)

        self.lbl_status.config(text=f"Đã lọc Butterworth (D0={radius}, n={n})", fg="green")



    def run_gaussian_highpass(self):

        if self.img_origin is None:

            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")

            return

        radius = self.slider_radius.get()

        self.lbl_status.config(text="Đang chạy Gaussian Highpass...", fg="blue")

        self.root.update()



        img_out, _, spec_filt, _ = gaussian_highpass_filter(self.img_origin, radius=radius)

        self.img_filtered = img_out

       

        self.show_image(spec_filt, self.lbl_img3)

        self.show_image(img_out, self.lbl_img4)

        self.lbl_status.config(text=f"Đã lọc Gaussian Highpass (D0={radius})", fg="green")



    def run_make_blur(self):

        if self.img_origin is None:

            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước!")

            return

        L = self.slider_L.get()

        sigma = self.slider_sigma.get()

       

        img_deg_float, img_deg_uint8, F_psf = apply_motion_blur(self.img_origin, length=L, noise_sigma=sigma)

        self.img_degraded_float = img_deg_float

        self.F_psf = F_psf

       

        # Cập nhật ảnh lỗi đè vào Khung gốc Ô (0,0) để làm Wiener

        self.show_image(img_deg_uint8, self.lbl_img1)

        self.lbl_status.config(text=f"Đã tạo ảnh lỗi: Nhòe L={L} + Nhiễu σ={sigma}", fg="red")



    def run_wiener(self):

        if self.img_origin is None or self.img_degraded_float is None or self.F_psf is None:

            messagebox.showwarning("Cảnh báo", "Bạn phải bấm tạo ảnh lỗi ở Bước 1 trước!")

            return

        self.lbl_status.config(text="Đang quét tìm hằng số K tối ưu theo SSIM...", fg="blue")

        self.root.update()



        K_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]

        best_img, opt_K, max_ssim = wiener_deconvolution(self.img_degraded_float, self.F_psf, self.img_origin, K_candidates=K_list)

        self.img_filtered = best_img

       

        self.show_image(best_img, self.lbl_img4)

       

        spec_filt = np.log(1 + np.abs(self.F_psf))

        spec_filt_uint8 = cv2.normalize(spec_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.show_image(spec_filt_uint8, self.lbl_img3)

        self.lbl_status.config(text=f"Khôi phục xong! K = {opt_K} | SSIM = {max_ssim:.4f}", fg="green")





    # ---- ĐIỀU KHIỂN HOẠT ĐỘNG VIDEO VÀ WEBCAM (SẮP XẾP VÀO LƯỚI 2X2) ----

    def load_video_file(self):

        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

        if file_path:

            if self.is_playing_video:

                self.toggle_video_flow()

            self.video_source = file_path

            self.lbl_status.config(text=f"Đã chọn tệp video: {os.path.basename(file_path)}", fg="green")

            self.btn_video.config(text="▶ Chạy Phân Tích Video", bg="#2196F3")

           

    def toggle_video_flow(self):

        if not self.is_playing_video:

            source = getattr(self, 'video_source', 0)

            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():

                messagebox.showerror("Lỗi", "Không thể kết nối nguồn Video/Webcam!")

                return

           

            ret, frame = self.cap.read()

            if not ret:

                self.cap.release()

                return

               

            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.prev_gray = cv2.resize(self.prev_gray, (320, 240))

           

            self.is_playing_video = True

            self.btn_video.config(text="⏹ Dừng phân tích", bg="#f44336")

            self.lbl_status.config(text="Đang chạy phân tích Horn-Schunck...", fg="purple")

            self.update_video_frame()

        else:

            self.is_playing_video = False

            if self.cap:

                self.cap.release()

            self.video_source = 0

            self.btn_video.config(text="🎥 Bật Camera / Phân tích", bg="#E91E63")

            self.lbl_status.config(text="Đã dừng luồng video.", fg="blue")



    def update_video_frame(self):

        if not self.is_playing_video:

            return

           

        ret, frame = self.cap.read()

        if not ret:

            self.toggle_video_flow()

            return

           

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        curr_gray = cv2.resize(curr_gray, (320, 240))

       

        # Khử nhiễu hạt trước khi tính đạo hàm không gian thời gian

        prev_blur = cv2.GaussianBlur(self.prev_gray, (5, 5), 0)

        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

       

        alpha = self.slider_alpha.get()

        iters = self.slider_of_iters.get()

        thresh = self.slider_thresh.get()

        step = self.slider_step.get()

       

        # Tính Optical Flow trường vector

        u, v = horn_schunck_optical_flow(prev_blur, curr_blur, alpha=alpha, num_iters=iters)

        result_frame = self.draw_filtered_quiver(curr_gray, u, v, step=step, threshold=thresh)

       

        # ĐỔI BỐ CỤC THEO ĐÚNG CHECKLIST 2x2:

        # 1. Ảnh gốc / Khung hình đầu vào hiện tại đẩy vào Ô (0,0)

        self.show_image(curr_gray, self.lbl_img1)

       

        # 2. Phổ biên độ tần số hiện tại tính và đẩy vào Ô (0,1)

        F_curr = np.fft.fftshift(np.fft.fft2(curr_gray.astype(np.float64)))

        spec_curr = np.log(1 + np.abs(F_curr))

        spec_curr_uint8 = cv2.normalize(spec_curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.show_image(spec_curr_uint8, self.lbl_img2)

       

        # 3. Phổ sau khi lọc (Đối với video ta copy phổ sang làm minh chứng phản hồi động) đẩy vào Ô (1,0)

        self.show_image(spec_curr_uint8, self.lbl_img3)

       

        # 4. Sơ đồ mũi tên Vector dịch chuyển đẩy vào Ô (1,1)

        self.show_image(result_frame, self.lbl_img4)

       

        self.prev_gray = curr_gray

        self.root.after(12, self.update_video_frame)



    def draw_filtered_quiver(self, img, u, v, step, threshold):

        h, w = img.shape

        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for y in range(0, h, step):

            for x in range(0, w, step):

                fx = u[y, x]

                fy = v[y, x]

                magnitude = np.sqrt(fx**2 + fy**2)

                if magnitude > threshold:

                    start_point = (x, y)

                    end_point = (int(x + fx * 3), int(y + fy * 3))

                    cv2.arrowedLine(img_bgr, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

        return img_bgr



if __name__ == "__main__":

    root = tk.Tk()

    app = DSPApp(root)

    root.mainloop()

