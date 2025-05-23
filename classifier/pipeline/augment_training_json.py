import json
from pathlib import Path

def add_verified_to_json(train_json, verified_root, class_name):
    new_entries = []
    cafo_dir = Path(verified_root) / class_name / "CAFO"

    with open(train_json, 'r') as f:
        data = json.load(f)

    existing_paths = {entry["image_path"] for entry in data}

    for img_path in cafo_dir.glob("*"):
        full_path = str(img_path.resolve())
        if full_path not in existing_paths:
            new_entries.append({
                "image_path": full_path,
                "label": 1,
                "split": "train"
            })

    if new_entries:
        print(f"✅ {len(new_entries)} new CAFOs added for {class_name}")
        data.extend(new_entries)
        with open(train_json, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        print(f"✅ No new CAFOs found for {class_name}")

    return len(new_entries)