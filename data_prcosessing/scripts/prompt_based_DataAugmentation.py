import ast
import itertools
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
import random
from tqdm import tqdm
import torch
from diffusers import StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# === Load Inpainting Pipeline ===
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# === Settings ===
df = pd.read_csv('/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/augmented_data/prompt_based_barn_location_cafo_with_boxes.csv')
n_samples = 5
prompt_pool = [
    "large trees",
    "a small black water pool",
    "grassland",
    "a circular shape pool",
    "small blue color house with roof"
]

output_dir = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/augmented_data_metainfo"
image_only_dir = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/augmented_data_image"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_only_dir, exist_ok=True)

meta_records = []
pdf_path = f"{output_dir}/all_inpainted_results.pdf"
pdf = PdfPages(pdf_path)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        detected_boxes = ast.literal_eval(row['detected_boxes'])
    except (ValueError, SyntaxError):
        continue
    if len(detected_boxes) == 0:
        continue

    fname = row['patch']
    try:
        with rasterio.open(fname) as src:
            red, green, blue = src.read(1), src.read(2), src.read(3)
            rgb_img = np.stack([red, green, blue], axis=-1).astype(np.float32)
            image_np = (rgb_img).astype(np.uint8)
            image_pil_orig = Image.fromarray(image_np)
            image_shape = image_np.shape[:2]  # (H, W)
    except Exception as e:
        print(f"[{idx}] Failed to load image: {fname} | Error: {e}")
        continue

    all_combinations = []
    for r in range(1, len(detected_boxes) + 1):
        all_combinations.extend(itertools.combinations(detected_boxes, r))
    sampled_combinations = random.sample(all_combinations, min(n_samples, len(all_combinations)))

    for i, combo in enumerate(sampled_combinations):
        mask = np.zeros(image_shape, dtype=np.uint8)
        for box in combo:
            x0, y0, x1, y1 = map(int, box)
            x0, x1 = max(0, x0), min(image_shape[1], x1)
            y0, y1 = max(0, y0), min(image_shape[0], y1)
            mask[y0:y1, x0:x1] = 1

        image_resized = image_pil_orig.resize((512, 512), Image.BILINEAR)
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)

        prompt = random.choice(prompt_pool)

        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                image=image_resized,
                mask_image=mask_resized,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.manual_seed(42)
            ).images[0]

        result_resized = result.resize(image_pil_orig.size, Image.BILINEAR)

        # Save only inpainted image
        gen_img_path = f"{image_only_dir}/gen_{idx}_combo{i}.png"
        result_resized.save(gen_img_path)

        # Save metadata
        meta_records.append({
            "id": idx,
            "combo_index": i,
            "patch": fname,
            "prompt": prompt,
            "inpainted_image_path": gen_img_path
        })

        # Side-by-side for global PDF
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_pil_orig)
        axs[0].set_title("Original")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Mask")
        axs[2].imshow(result_resized)
        axs[2].set_title(f"Inpainted: {prompt}")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# === Finalize ===
pdf.close()
print(f"Saved global PDF: {pdf_path}")

meta_df = pd.DataFrame(meta_records)
meta_df.to_csv(f"{output_dir}/inpainting_metadata.csv", index=False)
print(f"Metadata saved to: {output_dir}/inpainting_metadata.csv")
