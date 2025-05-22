import os
import json
import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import json
import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image

class JSONImageDataset(Dataset):
    def __init__(self, json_path, split="train", transform=None):
        with open(json_path) as f:
            all_data = json.load(f)

        # self.data = [item for item in all_data if item.get("split") == split]
        self.data = all_data[split]
        self.transform = transform

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Failed to load: {item['image_path']}")

        image = self.transform(image) if self.transform else image
        label = item["label"]
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.data)


class CAFODataModule(pl.LightningDataModule):
    def __init__(self, data_path="data", batch_size=16, mode="folder", worker=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.worker = worker

    def prepare_data(self):
        if self.mode == "folder":
            datasets.ImageFolder(self.data_path)

    def setup(self, stage=None):
        if self.mode == "folder":
            dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
            self.class_names = dataset.classes

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_set, self.val_set = random_split(dataset, [train_size, val_size])
        elif self.mode == "json":
            self.train_set = JSONImageDataset(self.data_path, split="train", transform=self.transform)
            self.val_set = JSONImageDataset(self.data_path, split="val", transform=self.transform)
            self.test_set = JSONImageDataset(self.data_path, split="test", transform=self.transform)
            self.class_names = ["Negative", "Swine", "Dairy", "Beef", "Poultry", "Horses", "Sheep/Goats"]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.worker, shuffle=True,  pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.worker, shuffle=False)

    def test_dataloader(self):
        # return self.val_dataloader()
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.worker, shuffle=False)

