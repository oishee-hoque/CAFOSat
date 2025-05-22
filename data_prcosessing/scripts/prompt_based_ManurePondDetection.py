import rasterio
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pandas as pd
from tqdm import tqdm

# === GroundingDINO Config ===
device = "cuda"
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# === Detection Prompts ===
manure_pond_prompts = [
    "a dark brown or black irregular pond surrounded by open land",
    "a small black water pool next to a large building",
    "a dark circular or rectangular patch in a cleared area",
    "a dark green or brown pool of water near agricultural buildings",
    "a small flat black or brown pit in open space",
    "a dirty water pond placed in a fenced or open farm area",
    "a rectangular dark spot with no reflection near farm infrastructure",
    "a shallow pit with dark liquid isolated from buildings",
    "a muddy or black spot on the ground surrounded by bare land",
    "a dark patch of water placed away from houses or roads"
]

# === I/O Paths ===
csv_path = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/combined_verified_cafo.csv"
output_csv = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/augmented_data/prompt_based_manure_pond_location_cafo_with_boxes.csv"

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Pre-fill Output Lists ===
boxes_list = [[] for _ in range(len(df))]
scores_list = [[] for _ in range(len(df))]
labels_list = [[] for _ in range(len(df))]

# === Process Each Patch ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    fname = row['patch']
    if row['manure_pond'] <= 0:
        continue

    try:
        with rasterio.open(fname) as src:
            red, green, blue = src.read(1), src.read(2), src.read(3)
            rgb_img = np.stack([red, green, blue], axis=-1).astype(np.float32)
            image_rgb = (rgb_img / np.max(rgb_img)).clip(0, 1)
            image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))

        # Model inference
        inputs = processor(images=image_pil, text=manure_pond_prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process predictions
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=0.35, text_threshold=0.25,
            target_sizes=[(image_pil.height, image_pil.width)]
        )[0]

        # Assign to pre-filled lists
        boxes_list[idx] = results["boxes"].cpu().numpy().tolist()
        scores_list[idx] = results["scores"].cpu().numpy().tolist()
        labels_list[idx] = results["text_labels"]

    except Exception as e:
        print(f"[{idx}] Failed on: {fname} | Error: {e}")

# === Save to CSV ===
df["detected_boxes"] = boxes_list
df["detected_scores"] = scores_list
df["detected_labels"] = labels_list
df.to_csv(output_csv, index=False)
print(f"Detection metadata saved to: {output_csv}")
