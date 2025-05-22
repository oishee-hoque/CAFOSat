import os
import torch
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/')
from model.classifierModel import CAFOClassifier

# # ==== CONFIG ====
# CONFIG = {
#     'checkpoint_path': '/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/checkpoints/cafo-best-epoch=02-val_acc=0.99_resnet50_IOWA.ckpt',
#     'image_folder': '/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/point_single_patches/DE_filtered',  # folder with .tif patches (assumed CAFO)
#     'output_csv': '/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/misclassified_cafos/filtered_misclassified_patches_DE.csv',
#     'output_csv_classified': '/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/misclassified_cafos/filtered_classified_patches_DE.csv',
#     'input_size': (224, 224),
#     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#     'cafo_class_id': 1
# }

# ==== Load Model ====
def load_model(CONFIG):
    model = CAFOClassifier.load_from_checkpoint(CONFIG['checkpoint_path'])
    model.eval().to(CONFIG['device'])
    return model


def read_patch_image(path):
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).astype(np.float32) / 255.0
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC -> PIL
    # print(img.shape)
    return Image.fromarray((img * 255).astype(np.uint8))

# ==== Main Inference ====
def predict_folder(CONFIG):
    model = load_model(CONFIG)
    all_files = [f for f in os.listdir(CONFIG['image_folder']) if f.endswith('.tif')]
    misclassified = []
    classified = []
    correct = 0
    
    # ==== Image Preprocessing ====
    transform = transforms.Compose([
        transforms.Resize(CONFIG['input_size']),
        transforms.ToTensor(),
    ])


    for fname in all_files:
        path = os.path.join(CONFIG['image_folder'], fname)
        try:
            img = read_patch_image(path)
            x = transform(img).unsqueeze(0).to(CONFIG['device'])
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()

            if pred == CONFIG['cafo_class_id']:
                correct += 1
                classified.append(fname)
            else:
                misclassified.append(fname)

        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    # === Accuracy ===
    total = len(all_files)
    accuracy = correct / total if total > 0 else 0.0

    print(f"Accuracy on CAFO folder: {accuracy*100:.2f}% ({correct}/{total})")
    
    # Save misclassified filenames
    df = pd.DataFrame({'misclassified_patch': misclassified})
    df.to_csv(CONFIG['output_csv'], index=False)
    df = pd.DataFrame({'classified_patch': classified})
    df.to_csv(CONFIG['output_csv_classified'], index=False)
    print(f"Saved non-CAFO predictions to: {CONFIG['output_csv']}")

# if __name__ == "__main__":
#     predict_folder()
