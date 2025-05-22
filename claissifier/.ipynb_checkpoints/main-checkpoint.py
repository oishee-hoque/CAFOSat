# main.py

import yaml
import os
from pathlib import Path
import csv
import json
import pandas as pd
from model.classifierModel import CAFOClassifier
from datas.datamodule import JSONImageDataset
from classify.classifier import CafoBinaryClassifier
from datas.datamodule import CAFODataModule
from pipeline.augment_training_json import add_verified_to_json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

def train_model(config, checkpoint_path=None):
    dm = CAFODataModule(**config["data"])
    model = CAFOClassifier(**config["model"])

    callbacks = [EarlyStopping(monitor="val_acc", mode="max", patience=config["trainer"].get("patience", 2))]
    
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        from pytorch_lightning.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(dirpath=os.path.dirname(checkpoint_path), filename=os.path.basename(checkpoint_path).replace('.ckpt', ''), save_top_k=1, monitor="val_acc", mode="max"))

    trainer = Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"].get("accelerator", "auto"),
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=dm)
    val_acc = trainer.callback_metrics.get("val_acc", 0.0).item()
    if checkpoint_path:
        trainer.save_checkpoint(checkpoint_path)
    return model, val_acc

def log_round(log_path, round_num, ckpt_name, new_count, val_acc):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Checkpoint", "New_CAFOs", "Val_Acc"])
        writer.writerow([round_num, ckpt_name, new_count, round(val_acc, 4)])

def active_loop(config):
    cfg = config
    round_num = 0
    Path(cfg["checkpoint"]["dirpath"]).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    while round_num < cfg["trainer"]["max_rounds"]:
        round_num += 1
        ckpt_name = f"round_{round_num}.ckpt"
        ckpt_path = os.path.join(cfg["checkpoint"]["dirpath"], ckpt_name)

        model, val_acc = train_model(config, checkpoint_path=ckpt_path)

        classifier = CafoBinaryClassifier(checkpoint_path=ckpt_path)

        if round_num == 1:
            classifier.classify_categories(
                base_input_dir=cfg["inference"]["input_dir"],
                output_base_dir=cfg["inference"]["verified_output"],
                categories=cfg["inference"]["categories"]
            )

        total_new = 0
        added_this_round = []

        for cat in cfg["inference"]["categories"]:
            total_new += add_verified_to_json(cfg["data"]["data_path"], cfg["inference"]["verified_output"], cat)
            newly_reclassed = classifier.reclassify_non_cafo(cfg["inference"]["verified_output"], cat)

            with open(cfg["data"]["data_path"], 'r') as f:
                data = json.load(f)

            for img_path in newly_reclassed:
                if not any(entry["image_path"] == img_path for entry in data):
                    data.append({"image_path": img_path, "label": 1, "split": "train"})
                    total_new += 1
                    added_this_round.append({
                        "round": round_num,
                        "image_path": img_path,
                        "source": cat,
                        "reason": "reclassified"
                    })

            with open(cfg["data"]["data_path"], 'w') as f:
                json.dump(data, f, indent=2)
        Path("logs/cafo_preds").mkdir(exist_ok=True)
        # Save log of added images
        if added_this_round:
            print("Hello")
            log_path = f"logs/cafo_preds/cafo_additions_round_{round_num}.csv"
            pd.DataFrame(added_this_round).to_csv(log_path, index=False)

        log_round(cfg["logging"]["csv_log"], round_num, ckpt_name, total_new, val_acc)

        if total_new == 0:
            print("✅ No new CAFOs identified this round. Stopping.")
            break

if __name__ == "__main__":
    with open("config/active_learning_config.yaml") as f:
        config = yaml.safe_load(f)
    active_loop(config)
