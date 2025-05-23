import yaml
import os
from model.classifierModel import CAFOClassifier
from datas.datamodule import CAFODataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main(config_file):
    with open(config_file) as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="config/classifier_config_iowa.yaml")
    args = parser.parse_args()
    main(config_file=args.config_file)
