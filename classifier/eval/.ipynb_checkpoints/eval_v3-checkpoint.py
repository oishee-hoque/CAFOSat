import os
import json
import torch
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from model.classifierModel import CAFOClassifier


model = "vit_b_16"
# === CONFIG ===
CONFIG = {
    'checkpoint_path': f'/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/checkpoints/cafo_binaray_mml_{model}.ckpt',
    'input_size': (224, 224),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

def load_model(config):
    model = CAFOClassifier.load_from_checkpoint(config['checkpoint_path'])
    model.eval().to(config['device'])
    return model

def read_patch_image(path):
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).astype(np.float32) / 255.0
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return Image.fromarray((img * 255).astype(np.uint8))

def evaluate_dataset(dataset, model, config):
    transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.ToTensor(),
    ])

    true_labels, pred_labels, failed = [], [], []

    for entry in dataset:
        patch_path = entry['image_path']
        # label = entry['label'] 1 if entry['label']>=1
        label = 1 if entry['label'] >= 1 else 0

        try:
            img = read_patch_image(patch_path)
            x = transform(img).unsqueeze(0).to(config['device'])
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()
            true_labels.append(label)
            pred_labels.append(pred)
        except Exception as e:
            print(f"⚠️ Failed to process {patch_path}: {e}")
            failed.append(patch_path)

    return true_labels, pred_labels, failed

def evaluate_json_dataset(json_path, config, label=""):
    with open(json_path, 'r') as f:
        data = json.load(f)

    model = load_model(config)

    all_data = []
    for split in ['train','val','test']:  # or just the ones you have
        if split in data:
            print(f"📦 Found split: {split} with {len(data[split])} samples.")
            all_data.extend(data[split])

    # Now evaluate all combined
    true_labels, pred_labels, failed = evaluate_dataset(all_data, model, config)

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    summary = {
        "split": "combined",
        "accuracy": np.mean(np.array(true_labels) == np.array(pred_labels)),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "failed_samples": failed
    }

    # Save results
    summary_path = f"results/evaluation_summary_combined_{label}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Combined evaluation summary saved to {summary_path}")   
    print(f"📊 Accuracy: {summary['accuracy']*100:.2f}% | Failed: {len(failed)} samples")

# === Example usage ===
evaluate_json_dataset(
    json_path="/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/cafo_set1.json",
    # json_path = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/nc_training_data.json",
    # json_path = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/train_test_split_meter_ml.json",
    config=CONFIG,
    label=f"mml_main_afo_{model}_verified"
)
