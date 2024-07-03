import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr_ssim_y_channel(original_image_path, reconstructed_image_path):
    # 读取原始图像和重建图像
    original_image = cv2.imread(original_image_path)
    reconstructed_image = cv2.imread(reconstructed_image_path)

    # 将图像从BGR转换为YCbCr
    original_ycbcr = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCR_CB)
    reconstructed_ycbcr = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2YCR_CB)

    # 提取Y通道
    original_y = original_ycbcr[:, :, 0]
    reconstructed_y = reconstructed_ycbcr[:, :, 0]

    # 计算PSNR
    psnr = peak_signal_noise_ratio(original_y, reconstructed_y)

    # 计算SSIM
    ssim = structural_similarity(original_y, reconstructed_y)

    return psnr, ssim

# 示例路径
original_image_path = 'eye_hr.png'
reconstructed_image_path = 'eye_LSRGAN_x2.png'

psnr, ssim = calculate_psnr_ssim_y_channel(original_image_path, reconstructed_image_path)
print(f'PSNR (Y channel): {psnr:.2f} dB')
print(f'SSIM (Y channel): {ssim:.4f}')