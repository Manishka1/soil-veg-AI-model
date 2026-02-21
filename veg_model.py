import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# -----------------------------
# Simple UNet Lite Architecture
# -----------------------------
class UNetLite(nn.Module):
    def __init__(self):
        super(UNetLite, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool(x1)
        x3 = self.encoder2(x2)
        x4 = self.up(x3)
        out = self.decoder(x4)
        return out


# -----------------------------
# Load Model
# -----------------------------
def load_veg_model():

    model = UNetLite()

    state_dict = torch.load(
        "models/veg_unet_lite.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_veg_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


# -----------------------------
# Predict
# -----------------------------
def predict_vegetation(model, image):
    input_tensor = preprocess_veg_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

    mask = output.squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)

    vegetation_percentage = (
        np.sum(binary_mask) / binary_mask.size * 100
    )

    mask_image = Image.fromarray(
        (binary_mask * 255).astype(np.uint8)
    )

    return mask_image, vegetation_percentage
