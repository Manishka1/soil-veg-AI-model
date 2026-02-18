import torch
from torchvision import transforms

# Change to your actual soil classes
SOIL_CLASSES = ["Clay", "Sandy", "Loamy", "Black Soil"]

def load_soil_model():
    model = torch.load(
        "models/soil_classifier.pth",
        map_location=torch.device("cpu")
    )
    model.eval()
    return model

def preprocess_soil_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image

