import os
import json
import random
from pathlib import Path

# === CONFIG ===
data_root = Path('/project/biocomplexity/gza5dr/CAFO/exp_v2/verified_cafo_dataset_main')  # Path to the root folder
train_ratio = 0.8  # 80% train, 20% val

# Define class mappings
binary_map = {
    'Beef_Cattle': 1,
    'Dairy_Cattle': 1,
    'Poultry': 1,
    'Swine': 1,
    'Negative': 0
}

multiclass_map = {
    'Beef_Cattle': 'cattle',
    'Dairy_Cattle': 'cattle',
    'Poultry': 'poultry',
    'Swine': 'swine',
    'Negative': 'negative'
}

def get_image_paths(folder):
    """Returns all image paths in a folder (non-recursive)."""
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ['.tif', '.png', '.jpg']]

def build_dataset(class_map):
    all_data = []

    for class_folder, label in class_map.items():
        folder_path = data_root / class_folder
        if not folder_path.exists():
            print(f"⚠️ Warning: Folder not found: {folder_path}")
            continue
        image_paths = get_image_paths(folder_path)
        all_data.extend([{"image_path": str(p), "label": label} for p in image_paths])

    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_ratio)
    return {
        "train": all_data[:split_idx],
        "test": all_data[split_idx:]
    }

# === Generate and Save ===
binary_dataset = build_dataset(binary_map)
multiclass_dataset = build_dataset(multiclass_map)

with open('dataset/binary_dataset.json', 'w') as f:
    json.dump(binary_dataset, f, indent=2)

with open('dataset/multiclass_dataset.json', 'w') as f:
    json.dump(multiclass_dataset, f, indent=2)

print("✅ JSON files created: binary_dataset.json, multiclass_dataset.json")
