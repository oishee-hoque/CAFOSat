import os
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # (1, C, H, W)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))
