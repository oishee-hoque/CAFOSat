import yaml
import os
from model.classifierModel import CAFOClassifier
from datas.datamodule import CAFODataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    with open("config/classifier_config_iowa.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Data & model
    dm = CAFODataModule(**config["data"])
    model = CAFOClassifier(**config["model"])
    
    os.makedirs(config["checkpoint"]["dirpath"], exist_ok=True)
    # Checkpoint from config
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    # Trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        **config["trainer"]
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
