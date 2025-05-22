import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import Accuracy

class CAFOClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, model_type="resnet18"):
        super().__init__()
        self.save_hyperparameters()

        if model_type == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        elif model_type == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        elif model_type == 'vit_b_16':
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

            # Safely get in_features from the final Linear layer inside the heads
            if isinstance(backbone.heads, nn.Sequential):
                in_features = backbone.heads[-1].in_features
            else:
                in_features = backbone.heads.in_features

            backbone.heads = nn.Linear(in_features, 2)
        elif model_type == "swin_b":
            backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            backbone.head = nn.Linear(backbone.head.in_features, 2)
        elif model_type == "convnext_base":
            backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, 2)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model = backbone
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
