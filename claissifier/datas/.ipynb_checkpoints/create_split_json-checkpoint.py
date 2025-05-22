import os
import json
import random
from pathlib import Path

def create_split_json(dataset_root, output_json, train_ratio=0.8, seed=42):
    label_map = {
        "cafo": 1,
        "notcafo": 0
    }

    entries = []
    random.seed(seed)

    for class_name, label in label_map.items():
        class_dir = Path(dataset_root) / class_name
        if not class_dir.exists():
            print(f"⚠️ Missing directory: {class_dir}")
            continue

        images = [img for img in class_dir.glob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png",'.tif']]
        random.shuffle(images)
        split_idx = int(train_ratio * len(images))

        for i, img in enumerate(images):
            split = "train" if i < split_idx else "test"
            entries.append({
                "image_path": str(img.resolve()),
                "label": label,
                "split": split
            })

    with open(output_json, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"✅ Saved {len(entries)} entries to {output_json}")
    print(f"📊 Train: {sum(e['split'] == 'train' for e in entries)}, Test: {sum(e['split'] == 'test' for e in entries)}")

if __name__ == "__main__":
    create_split_json(
        dataset_root="/project/biocomplexity/gza5dr/CAFO/NC_data/training_data",
        output_json="/project/biocomplexity/gza5dr/CAFO/NC_data/nc_training_data.json",
        train_ratio=0.8
    )
