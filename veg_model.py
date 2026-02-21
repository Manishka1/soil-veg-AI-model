import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# =====================================
# EXACT TRAINING ARCHITECTURE
# =====================================

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU(),
            nn.Conv2d(o, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNetLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.b = DoubleConv(128, 256)

        self.u3 = DoubleConv(256 + 128, 128)
        self.u2 = DoubleConv(128 + 64, 64)
        self.u1 = DoubleConv(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        b = self.b(self.pool(d3))

        x = F.interpolate(b, scale_factor=2)
        x = self.u3(torch.cat([x, d3], 1))

        x = F.interpolate(x, scale_factor=2)
        x = self.u2(torch.cat([x, d2], 1))

        x = F.interpolate(x, scale_factor=2)
        x = self.u1(torch.cat([x, d1], 1))

        # IMPORTANT: training used sigmoid here
        return torch.sigmoid(self.out(x))


# =====================================
# LOAD MODEL
# =====================================

def load_veg_model():

    model = UNetLite()

    state_dict = torch.load(
        "models/veg_unet_lite.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# =====================================
# PREPROCESS (MATCH TRAINING)
# =====================================

def preprocess_veg_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),   # scales to 0–1 (correct)
    ])
    return transform(image).unsqueeze(0)


# =====================================
# PREDICTION
# =====================================

def predict_vegetation(model, image):

    input_tensor = preprocess_veg_image(image)

    with torch.no_grad():
        output = model(input_tensor)  # sigmoid already applied

    mask = output.squeeze().cpu().numpy()

    # 🔥 Use calibrated threshold (0.3 works better for BCE+Dice)
    threshold = 0.3
    binary_mask = (mask > threshold).astype(np.uint8)

    vegetation_percentage = (
        np.sum(binary_mask) / binary_mask.size * 100
    )

    # Convert to displayable image
    mask_image = Image.fromarray(
        (binary_mask * 255).astype(np.uint8)
    )

    return mask_image, vegetation_percentage
