import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def load_veg_model():

    checkpoint = torch.load(
        "models/veg_unet_lite.pth",
        map_location="cpu",
        weights_only=False  # required for torch 2.6+
    )

    # If full model was saved
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint

    # If state_dict was saved
    elif isinstance(checkpoint, dict):
        # ⚠️ You must recreate architecture here
        # Replace this with your actual UNet class
        from your_unet_file import UNetLite   # <-- change this

        model = UNetLite()
        model.load_state_dict(checkpoint)

    else:
        raise ValueError("Unknown model format")

    model.eval()
    return model


def preprocess_veg_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


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
