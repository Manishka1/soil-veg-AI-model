import torch
from torchvision import transforms

SOIL_CLASSES = ["alluvial", "black", "clay", "red"]

def load_soil_model():

    # IMPORTANT: must disable weights_only in torch 2.6+
    model = torch.load(
        "models/soil_classifier.pth",
        map_location="cpu",
        weights_only=False
    )

    model.eval()
    return model


def preprocess_soil_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)