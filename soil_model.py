import torch
import torch.nn as nn
from torchvision import models, transforms

# IMPORTANT: must match your Roboflow class order
SOIL_CLASSES = ["alluvial", "black", "clay", "red"]

def load_soil_model():

    # Recreate MobileNetV3 exactly like training
    model = models.mobilenet_v3_small(weights=None)

    # Replace final classifier layer
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(SOIL_CLASSES)
    )

    # Load weights (state_dict)
    state_dict = torch.load(
        "models/soil_classifier.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

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
