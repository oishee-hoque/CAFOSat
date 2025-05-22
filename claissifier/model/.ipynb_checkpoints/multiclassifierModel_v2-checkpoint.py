import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAveragePrecision
import timm  # for DINOv2 etc.
import open_clip  # for CLIP and RemoteCLIP

class CAFOClassifier(pl.LightningModule):
    def __init__(self, num_classes=6, lr=1e-4, model_name='resnet18'):
        super().__init__()
        self.save_hyperparameters()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Linear(self.backbone.heads.in_features, num_classes)

        elif model_name == 'swin_b':
            self.backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

        elif model_name == 'dinov2_vit_b':
            self.backbone = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
            self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

        elif model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone.classifier[2] = nn.Linear(self.backbone.classifier[2].in_features, num_classes)

        elif model_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            self.clip_image_encoder = model.visual
            self.backbone = nn.Sequential(
                self.clip_image_encoder,
                nn.Linear(model.visual.output_dim, num_classes)
            )

        elif model_name == 'remoteclip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='remoteclip_S2')
            self.clip_image_encoder = model.visual
            self.backbone = nn.Sequential(
                self.clip_image_encoder,
                nn.Linear(model.visual.output_dim, num_classes)
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_map = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)
        map_score = self.val_map(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_map", map_score, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
