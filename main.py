import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Label, Button, Radiobutton, IntVar
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from model_SRCNN import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
import cv2
from restinal import *
import random
import os
import cv2
import torch
from torchvision import transforms
import lsrgan_config
import model_LSRGAN
from utils import make_directory

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1000x800")

        # 初始化图像显示区域
        self.original_image = None
        self.processed_image = None
        self.original_image_label = tk.Label(self.root)
        self.processed_image_label = tk.Label(self.root)

        # 创建按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=20)

        # 创建按钮
        self.btn_open = tk.Button(button_frame, text="Open Image", command=self.open_image)
        self.btn_reset = tk.Button(button_frame, text="Reset", command=self.reset_images)
        self.btn_save = tk.Button(button_frame, text="Save Image", command=self.save_image)
        self.btn_enhance = tk.Button(button_frame, text="Enhance", command=self.show_custom_options2)
        self.btn_add_noise = tk.Button(button_frame, text="Add Noise", command=self.show_custom_options3)
        self.btn_edge_detection = tk.Button(button_frame, text="Edge Detection", command=self.show_custom_options)
        self.btn_sr = tk.Button(button_frame, text="Super Resolution", command=self.show_super_resolution_options)
        self.btn_restinal = tk.Button(button_frame, text="Detect Blood Vessels", command=self.restinal)
        self.btn_noise_reduction = tk.Button(button_frame, text="Noise Reduction", command=self.show_noise_reduction_options)


        # 布局按钮
        self.btn_open.grid(row=0, column=0, padx=10, pady=5)
        self.btn_reset.grid(row=0, column=1, padx=10, pady=5)
        self.btn_save.grid(row=0, column=2, padx=10, pady=5)
        self.btn_enhance.grid(row=0, column=3, padx=10, pady=5)
        self.btn_add_noise.grid(row=0, column=4, padx=10, pady=5)
        self.btn_edge_detection.grid(row=0, column=5, padx=10, pady=5)
        self.btn_sr.grid(row=0, column=6, padx=10, pady=5)
        self.btn_restinal.grid(row=0, column=7, padx=10, pady=5)
        self.btn_noise_reduction.grid(row=0, column=8, padx=10, pady=5)


        # 布局图像标签
        self.original_image_label.pack(side=tk.LEFT, padx=20, pady=20)
        self.processed_image_label.pack(side=tk.RIGHT, padx=20, pady=20)

    def show_super_resolution_options(self):
        sr_window = Toplevel(self.root)
        sr_window.title("Super Resolution Options")

        Label(sr_window, text="Choose a super resolution method:").pack(pady=10)

        self.sr_method = IntVar()
        self.sr_method.set(1)

        Radiobutton(sr_window, text="Bicubic", variable=self.sr_method, value=1).pack(anchor=tk.W)
        Radiobutton(sr_window, text="SRCNN", variable=self.sr_method, value=2).pack(anchor=tk.W)
        Radiobutton(sr_window, text="LSRGAN", variable=self.sr_method, value=3).pack(anchor=tk.W)

        Button(sr_window, text="Apply", command=self.apply_super_resolution).pack(pady=20)

    def apply_super_resolution(self):
        method = self.sr_method.get()
        if method == 1:
            self.upscale_image_with_bicubic()
        elif method == 2:
            self.super_resolution()
        elif method == 3:
            self.LSRGAN_SR()

    def upscale_image_with_bicubic(self, scale_factor=2):
        if self.original_image:
            upscaled_image = self.processed_image.resize(
                (self.processed_image.width * scale_factor, self.processed_image.height * scale_factor),
                resample=Image.BICUBIC)
            self.processed_image = upscaled_image
            self.update_images()

    def super_resolution(self, weights='SRCNN_x2.pth'):
        try:
            weights_file = weights
            image = self.processed_image
            scale = 2

            cudnn.benchmark = True
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            model = SRCNN().to(device)

            state_dict = model.state_dict()
            for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(n)

            model.eval()

            if not isinstance(image, pil_image.Image):
                raise ValueError("self.processed_image is not a valid PIL Image")

            image_width = (image.width // scale) * scale
            image_height = (image.height // scale) * scale
            image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
            image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

            image = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image)

            y = ycbcr[..., 0]
            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)

            self.processed_image = output
            self.update_images()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def open_image(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File",
                                              filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"),
                                                         ("All files", "*.*")))
        if filename:
            try:
                self.original_image = Image.open(filename)
                self.processed_image = self.original_image.copy()
                self.update_images()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def show_custom_options(self):
        edge_detection_window = Toplevel(self.root)
        edge_detection_window.title("Edge Detection Options")

        Label(edge_detection_window, text="Choose an edge detection method:").pack(pady=10)

        self.edge_detection_method = IntVar()
        self.edge_detection_method.set(1)

        Radiobutton(edge_detection_window, text="Sobel", variable=self.edge_detection_method, value=1).pack(anchor=tk.W)
        Radiobutton(edge_detection_window, text="Robert", variable=self.edge_detection_method, value=2).pack(anchor=tk.W)
        Radiobutton(edge_detection_window, text="Laplacian", variable=self.edge_detection_method, value=3).pack(anchor=tk.W)

        Button(edge_detection_window, text="Apply", command=self.apply_edge_detection).pack(pady=20)

    def apply_edge_detection(self):
        method = self.edge_detection_method.get()
        if self.original_image:
            if method == 1:
                self.sobel_operator()
            elif method == 2:
                self.ropert_operator()
            elif method == 3:
                self.laplacian_operator()

    def sobel_operator(self):
        if self.original_image:
            img_np = np.array(self.original_image)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_edges = cv2.magnitude(sobelx, sobely)
            sobel_edges = np.uint8(255 * sobel_edges / np.max(sobel_edges))
            self.processed_image = Image.fromarray(cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2RGB))
            self.update_images()

    def ropert_operator(self):
        if self.original_image:
            img_np = np.array(self.original_image)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
            kernely = np.array([[0, -1], [1, 0]], dtype=int)
            x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
            y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.processed_image = Image.fromarray(cv2.cvtColor(Roberts, cv2.COLOR_GRAY2RGB))
            self.update_images()

    def laplacian_operator(self):
        if self.original_image:
            img_np = np.array(self.original_image)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            grayImage = cv2.GaussianBlur(img_gray, (5, 5), 0, 0)
            dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
            Laplacian = cv2.convertScaleAbs(dst)
            self.processed_image = Image.fromarray(cv2.cvtColor(Laplacian, cv2.COLOR_GRAY2RGB))
            self.update_images()

    def update_images(self):
        if self.original_image:
            self.original_image_tk = ImageTk.PhotoImage(self.original_image)
            self.processed_image_tk = ImageTk.PhotoImage(self.processed_image)
            self.original_image_label.config(image=self.original_image_tk)
            self.original_image_label.image = self.original_image_tk
            self.processed_image_label.config(image=self.processed_image_tk)
            self.processed_image_label.image = self.processed_image_tk

    def show_custom_options2(self):
        enhance_window = Toplevel(self.root)
        enhance_window.title("Enhance Options")

        Label(enhance_window, text="Choose an enhancement method:").pack(pady=10)

        self.enhance_method = IntVar()
        self.enhance_method.set(1)

        Radiobutton(enhance_window, text="Brightness Enhancement", variable=self.enhance_method, value=1).pack(anchor=tk.W)
        Radiobutton(enhance_window, text="Histogram Equalization", variable=self.enhance_method, value=2).pack(anchor=tk.W)

        Button(enhance_window, text="Apply", command=self.apply_enhancement).pack(pady=20)

    def apply_enhancement(self):
        method = self.enhance_method.get()
        if self.original_image:
            if method == 1:
                self.enhance_image()
            elif method == 2:
                self.balance_image()

    def balance_image(self): # 直方图均衡化
        if self.original_image:
            img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(img_gray)
            self.processed_image = Image.fromarray(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
            self.update_images()

    def enhance_image(self):
        if self.original_image:
            enhancer = ImageEnhance.Contrast(self.processed_image)
            self.processed_image = enhancer.enhance(1.5)
            self.update_images()

    def show_custom_options3(self):
        noise_window = Toplevel(self.root)
        noise_window.title("Add Noise Options")

        Label(noise_window, text="Choose a noise type:").pack(pady=10)

        self.noise_type = IntVar()
        self.noise_type.set(1)

        Radiobutton(noise_window, text="Gaussian Noise", variable=self.noise_type, value=1).pack(anchor=tk.W)
        Radiobutton(noise_window, text="Salt and Pepper Noise", variable=self.noise_type, value=2).pack(anchor=tk.W)

        Button(noise_window, text="Apply", command=self.apply_noise).pack(pady=20)

    def apply_noise(self):
        noise_type = self.noise_type.get()
        if noise_type == 1:
            self.add_Gaussian_noise()
        elif noise_type == 2:
            self.add_salt_pepper_noise()

    def add_Gaussian_noise(self):
        if self.original_image:
            img_np = np.array(self.original_image)
            mean = 0
            var = 0.01
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, img_np.shape)
            noisy = img_np + gauss
            self.processed_image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
            self.update_images()

    def add_salt_pepper_noise(self):
        if self.original_image:
            img_np = np.array(self.original_image)
            prob = 0.05
            noisy = img_np.copy()
            black = np.random.rand(*img_np.shape[:2]) < prob
            white = np.random.rand(*img_np.shape[:2]) < prob
            noisy[black] = 0
            noisy[white] = 255
            self.processed_image = Image.fromarray(noisy)
            self.update_images()

    def show_noise_reduction_options(self):
        noise_reduction_window = Toplevel(self.root)
        noise_reduction_window.title("Noise Reduction Options")

        Label(noise_reduction_window, text="Choose a noise reduction method:").pack(pady=10)

        self.noise_reduction_method = IntVar()
        self.noise_reduction_method.set(1)

        Radiobutton(noise_reduction_window, text="Mean Filter", variable=self.noise_reduction_method, value=1).pack(anchor=tk.W)
        Radiobutton(noise_reduction_window, text="Median Filter", variable=self.noise_reduction_method, value=2).pack(anchor=tk.W)
        Radiobutton(noise_reduction_window, text="Bilateral Filter", variable=self.noise_reduction_method, value=3).pack(anchor=tk.W)

        Button(noise_reduction_window, text="Apply", command=self.apply_noise_reduction).pack(pady=20)

    def apply_noise_reduction(self):
        method = self.noise_reduction_method.get()
        if self.original_image:
            img_np = np.array(self.original_image)
            if method == 1:
                result = cv2.blur(img_np, (5, 5))
            elif method == 2:
                result = cv2.medianBlur(img_np, 5)
            elif method == 3:
                result = cv2.bilateralFilter(img_np, 9, 75, 75)
            self.processed_image = Image.fromarray(result)
            self.update_images()

    def edge_detection(self):
        if self.original_image:
            self.processed_image = self.processed_image.filter(ImageFilter.FIND_EDGES)
            self.update_images()

    def reset_images(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.update_images()

    def save_image(self):
        if self.processed_image:
            try:
                filetypes = (("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
                filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
                if filepath:
                    self.processed_image.save(filepath)
                    messagebox.showinfo("Image Saved", f"Image saved successfully at {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def restinal(self):
        try:
            srcImg = np.array(self.processed_image)
            grayImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2GRAY)
            ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(th0, np.ones((7, 7), np.uint8))
            blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
            heImg = cv2.equalizeHist(blurImg)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
            claheImg = clahe.apply(blurImg)
            homoImg = homofilter(blurImg)
            preMFImg = adjust_gamma(claheImg, gamma=1.5)
            filters = build_filters2()
            gaussMFImg = process(preMFImg, filters)
            gaussMFImg_mask = pass_mask(mask, gaussMFImg)
            grayStretchImg = grayStretch(gaussMFImg_mask, m=30.0 / 255, e=8)
            ret1, th1 = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_OTSU)
            predictImg = th1.copy()
            wtf = np.hstack([srcImg, cv2.cvtColor(predictImg, cv2.COLOR_GRAY2BGR)])
            wtf_pil = Image.fromarray(wtf)
            self.processed_image = wtf_pil
            self.update_images()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def LSRGAN_SR(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        msrn_model = model_LSRGAN.__dict__[lsrgan_config.g_arch_name](in_channels=lsrgan_config.in_channels,
                                                                      out_channels=lsrgan_config.out_channels,
                                                                      channels=lsrgan_config.channels,
                                                                      growth_channels=lsrgan_config.growth_channels,
                                                                      num_blocks=lsrgan_config.num_blocks)
        msrn_model = msrn_model.to(device=lsrgan_config.device)
       # print(f"Build `{lsrgan_config.g_arch_name}` model successfully.")

        # Load the super-resolution model weights
        checkpoint = torch.load(lsrgan_config.g_model_weights_path, map_location=lambda storage, loc: storage)
        msrn_model.load_state_dict(checkpoint["state_dict"])
        #print(f"Load `{lsrgan_config.g_arch_name}` model weights "
          #    f"`{os.path.abspath(lsrgan_config.g_model_weights_path)}` successfully.")

        # Create a folder of super-resolution experiment results
       # make_directory(lsrgan_config.sr_dir)

        # Start the verification mode of the model.
        msrn_model.eval()

        # Process the specified image
        lr_image = self.processed_image
        lr_image = np.array(lr_image)
        #print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Convert BGR image to RGB
        image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image to a PyTorch tensor
        transform = transforms.ToTensor()
        tensor = transform(image)

        # Add batch dimension and move to specified device
        tensor = tensor.unsqueeze(0).to(device)

        lr_tensor = tensor

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = msrn_model(lr_tensor)

        # Save image
        sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_image = (sr_image * 255.0).clip(0, 255).astype("uint8")
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

        # 将 numpy 数组转换为 PIL 图像
        pil_image = Image.fromarray(sr_image)
        self.processed_image = pil_image
        self.update_images()
        #print(f"Super-resolved image saved to `{os.path.abspath(sr_image_path)}`")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
