import torch
import yaml
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from model.multiclassifierModel import CAFOClassifier
from datas.datamodule_multiclass import CAFODataModule


def evaluate_model(model, dataloader, num_classes):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    for batch in dataloader:
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_logits.append(logits.cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    labels_np = np.array(all_labels)

    report = classification_report(labels_np, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(labels_np, all_preds)

    y_true_bin = label_binarize(labels_np, classes=list(range(num_classes)))
    map_score = average_precision_score(y_true_bin, logits_np, average="macro")

    return report, conf_matrix, map_score


def main(config_file):
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # === Data
    dm = CAFODataModule(
        data_path=cfg["data_path"],
        batch_size=cfg["batch_size"],
        mode="json",
        worker=cfg["worker"]
    )

    # === Model
    model = CAFOClassifier(
        num_classes=cfg["num_classes"],
        lr=cfg["lr"],
        model_name=cfg["model_name"]
    )

    # === Logger and Callbacks
    logger = CSVLogger(save_dir=cfg["log_dir"], name=f"{cfg['data_type']}_{cfg['model_name']}")
    checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True)
    earlystop_cb = EarlyStopping(monitor="val_acc", patience=cfg["patience"], mode="max")

    # === Trainer
    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        accelerator="gpu" if torch.cuda.is_available() and cfg["gpus"] > 0 else "cpu",
        devices=cfg["gpus"],
        precision=cfg["precision"],
        log_every_n_steps=2
    )

    # === Train
    trainer.fit(model, datamodule=dm)

    # === Evaluate on validation set
    val_report, val_conf_matrix, val_map = evaluate_model(model, dm.val_dataloader(), cfg["num_classes"])

    # === Evaluate on test set
    test_report, test_conf_matrix, test_map = evaluate_model(model, dm.test_dataloader(), cfg["num_classes"])

    # === Save all results
    results = {
        "model": cfg["model_name"],
        "best_val_acc": float(trainer.callback_metrics.get("val_acc", -1)),
        "best_val_f1": float(trainer.callback_metrics.get("val_f1", -1)),
        "log_dir": logger.log_dir,
        "checkpoint_path": checkpoint_cb.best_model_path,
        "val_classification_report": val_report,
        "val_confusion_matrix": val_conf_matrix.tolist(),
        "val_mAP": float(val_map),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_conf_matrix.tolist(),
        "test_mAP": float(test_map),
    }

    output_path = f"{cfg['data_type']}_{cfg['model_name']}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Full training results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="config/classifier_config_iowa.yaml")
    args = parser.parse_args()
    main(config_file=args.config_file)
