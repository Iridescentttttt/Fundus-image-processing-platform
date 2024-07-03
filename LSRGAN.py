# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import cv2
import torch
from torchvision import transforms
import lsrgan_config
import model_LSRGAN
from utils import make_directory

def preprocess_image(image_path, device):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to a PyTorch tensor
    transform = transforms.ToTensor()
    tensor = transform(image)

    # Add batch dimension and move to specified device
    tensor = tensor.unsqueeze(0).to(device)

    return tensor

def main() -> None:
    # Initialize the super-resolution model
    msrn_model = model_LSRGAN.__dict__[lsrgan_config.g_arch_name](in_channels=lsrgan_config.in_channels,
                                                           out_channels=lsrgan_config.out_channels,
                                                           channels=lsrgan_config.channels,
                                                           growth_channels=lsrgan_config.growth_channels,
                                                           num_blocks=lsrgan_config.num_blocks)
    msrn_model = msrn_model.to(device=lsrgan_config.device)
    print(f"Build `{lsrgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(lsrgan_config.g_model_weights_path, map_location=lambda storage, loc: storage)
    msrn_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{lsrgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(lsrgan_config.g_model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(lsrgan_config.sr_dir)

    # Start the verification mode of the model.
    msrn_model.eval()

    # Process the specified image
    lr_image_path = "eye_lr.jpg"
    sr_image_path = os.path.join(lsrgan_config.sr_dir, "eye_LSRGAN_x2.jpg")

    print(f"Processing `{os.path.abspath(lr_image_path)}`...")
    lr_tensor = preprocess_image(lr_image_path, lsrgan_config.device)

    # Only reconstruct the Y channel image data.
    with torch.no_grad():
        sr_tensor = msrn_model(lr_tensor)

    # Save image
    sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_image = (sr_image * 255.0).clip(0, 255).astype("uint8")
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sr_image_path, sr_image)
    print(f"Super-resolved image saved to `{os.path.abspath(sr_image_path)}`")

if __name__ == "__main__":
    main()
