import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch

def parse_flag(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.strip().lower()
        if val in ['yes', 'true']:
            return 1.0
        try:
            return 1.0 if float(val) != 0 else 0.0
        except ValueError:
            return 0.0
    if isinstance(val, (int, float)):
        return 1.0 if val != 0 else 0.0
    return 0.0

class CAFOCsvDataset(Dataset):
    def __init__(self, data_dir, df, mode='binary', transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_dir+row['patch_file']
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = int(row.get("label", 0))
        if self.mode == 'binary':
            label = int(label > 0)

        if self.mode == 'bbox' or self.mode == 'all':
            label = int(row.get("label", 0))

            if label > 0:
                bbox_val = row.get("geom_bbox", "[5.0, 5.0, 220.0, 220.0]")
                if isinstance(bbox_val, str):
                    try:
                        import ast
                        bbox = np.array(ast.literal_eval(bbox_val), dtype=np.float32)
                        # Scale from original image size to transformed size
                        original_size = 833
                        resized_size = 224  # .size returns (W, H)
                        scale = resized_size / original_size

                        bbox = np.array([coord * scale for coord in bbox], dtype=np.float32)
                    except:
                        bbox = np.array([5.0, 5.0, 220.0, 220.0], dtype=np.float32)
                else:
                    bbox = np.array([5.0, 5.0, 220.0, 220.0], dtype=np.float32)
            else:
                bbox = np.array([5.0, 5.0, 220.0, 220.0], dtype=np.float32)


        if self.mode == 'infra' or self.mode == 'all':
            infra = np.array([
               row.get("barn"),
               row.get("manure_pond"),
                row.get("grazing_area"),
               row.get("others")
            ], dtype=np.float32)

        if self.mode == 'bbox':
            return image, label, bbox
        elif self.mode == 'infra':
            return image, label, infra
        elif self.mode == 'all':
            return image, label, infra, bbox
        else:
            return image, label

def get_cafo_dataloader_from_csv(data_dir, csv_path, dataset_name='set1', split='train', task='binary',
                                  batch_size=32, shuffle=True, transform=None):
    """
    Args:
        csv_path: Path to the full metadata CSV
        dataset_name: one of set1, set2, merged, verified, etc.
        split: one of 'train', 'test', 'val'
        task: 'binary', 'bbox', 'infra', 'all'
    """
    df = pd.read_csv(csv_path)

    split_col = f"cafosat_{dataset_name}_training_{split}"
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' not found in CSV.")

    # Filter based on split membership
    df_split = df[df[split_col] == 1]
    df_split = df_split.dropna(subset=['label'])
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    dataset = CAFOCsvDataset(data_dir, df_split, mode=task, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def show_image_with_bbox(image_tensor, bbox, infra_tensor, label):
    # Convert image tensor to PIL for plotting
    image = F.to_pil_image(image_tensor)

    # Unpack bbox
    x_min, y_min, x_max, y_max = bbox.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.add_patch(plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    ))

    # Show infra flags
    labels = ["barn", "manure_pond", "grazing_area", "others"]
    flags = [int(x) for x in infra_tensor.tolist()]
    flag_str = "\n".join([f"{k}: {v}" for k, v in zip(labels, flags)])

    plt.title(flag_str)
    plt.axis('off')
    plt.show()
