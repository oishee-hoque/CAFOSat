import os
import json
import torch
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from torchvision import transforms
from model.classifierModel import CAFOClassifier


model = 'swin'
dtype = "mml"
# === CONFIG ===
CONFIG = {
    'checkpoint_path': f'/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/checkpoints/cafo_binaray_{dtype}_{model}.ckpt',
    'input_size': (224, 224),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'cafo_class_id': 1  # Assuming 1 means CAFO
}

# === Load Model ===
def load_model(config):
    model = CAFOClassifier.load_from_checkpoint(config['checkpoint_path'])
    model.eval().to(config['device'])
    return model

# === Read Image ===
def read_patch_image(path):
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).astype(np.float32) / 255.0
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return Image.fromarray((img * 255).astype(np.uint8))

def evaluate_json(json_path, config, summary_output_path, error_list_path):
    model = load_model(config)
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.ToTensor(),
    ])

    total = 0
    correct = 0
    misclassified = []

    for entry in data:
        patch_path = entry['patch_file']
        total += 1
        try:
            img = read_patch_image(patch_path)
            x = transform(img).unsqueeze(0).to(config['device'])
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()

            if pred == config['cafo_class_id']:
                correct += 1
            else:
                misclassified.append(patch_path)

        except Exception as e:
            print(f"⚠️ Failed to process {patch_path}: {e}")
            misclassified.append(patch_path)

    accuracy = correct / total if total > 0 else 0.0
    print(f"✅ {os.path.basename(json_path)}: {accuracy*100:.2f}% accuracy ({correct}/{total})")

    # Save summary JSON
    summary = {
        "json_file": os.path.basename(json_path),
        "total": total,
        "correct_cafo_predictions": correct,
        "missed_predictions": len(misclassified),
        "accuracy": round(accuracy, 4)
    }
    with open(summary_output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save misclassified patch paths
    with open(error_list_path, 'w') as f:
        json.dump(misclassified, f, indent=2)

    print(f"📄 Saved summary to: {summary_output_path}")
    print(f"🧾 Saved misclassified patch list to: {error_list_path}")

    
    
# === Run Evaluation on All JSON Files ===
json_files = {
    # "verified": "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/verified_cafo_patch_data.json",
    "main": "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/main_cafo_patch_data.json",
    "filtered": "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/filtered_cafo_patch_data.json"
}

for label, path in json_files.items():
    summary_json = f"eval/results/refine_coord_exp/evaluation_summary_{label}_{dtype}_{model}.json"
    error_list = f"eval/results/refine_coord_exp/misclassified_{label}_patches_{dtype}_{model}.json"
    evaluate_json(path, CONFIG, summary_json, error_list)
