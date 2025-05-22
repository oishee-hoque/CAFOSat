import os
import json
import glob
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split

# === Step 1: Load and filter labels ===
label_path = "/project/biocomplexity/wyr6fx(Nibir)/NeurIPS/Meter_ML/geofiles/train_dataset.geojson"
label_df = gpd.read_file(label_path)

# Normalize folder names for matching
label_df['Image_Folder'] = label_df['Image_Folder'].str.strip().str.lower()

# Keep only CAFO and Negative types
label_df = label_df[label_df['Type'].isin(['CAFOs', 'Negative'])]
if label_df.empty:
    raise ValueError("❌ No CAFO or Negative labels found in the dataset.")

label_df['Image_Folder'] = label_df['Image_Folder'].str.replace("train_images/", "", regex=False)
# Assign binary labels
label_df['label'] = label_df['Type'].apply(lambda x: 1 if x == 'CAFOs' else 0)
label_dict = dict(zip(label_df['Image_Folder'], label_df['label']))

print(f"✅ Found {len(label_df)} labeled CAFO/Negative entries")

# === Step 2: Find NAIP images ===
image_dirs = glob.glob("/project/biocomplexity/wyr6fx(Nibir)/NeurIPS/Meter_ML/geofiles/train_images_*/*/")
data_entries = []

for folder in image_dirs:
    folder = folder.rstrip("/")
    base_name = os.path.basename(folder).strip().lower()
    if base_name not in label_dict:
        continue

    # Find NAIP .png image
    naip_files = [f for f in os.listdir(folder) if "naip" in f.lower() and f.lower().endswith(".png")]
    naip_files = [os.path.join(folder, f) for f in naip_files]
    if not naip_files:
        print(f"⚠️ No NAIP .png found in folder: {folder}")
        continue

    image_path = naip_files[0]  # Use the first NAIP image
    label = label_dict[base_name]
    
    data_entries.append({
        "image_path": image_path,
        "label": label
    })

print(f"🧾 Collected {len(data_entries)} labeled NAIP image entries")

if len(data_entries) == 0:
    raise ValueError("❌ No valid labeled image entries were collected. Check folder names or NAIP file availability.")

# === Step 3: Train/test split ===
train, test = train_test_split(
    data_entries,
    test_size=0.1,
    stratify=[entry['label'] for entry in data_entries],
    random_state=42
)

# === Step 4: Save output ===
output = {"train": train, "test": test}
os.makedirs("dataset", exist_ok=True)
output_path = "dataset/train_test_split_meter_ml.json"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved train/test split with {len(train)} train and {len(test)} test samples to {output_path}")
