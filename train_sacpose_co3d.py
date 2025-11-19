import sys
import yaml
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from modules.sacpose_co3d import training

def main(cfg):
    cfg["RUN_NAME"] = 'SACPose_CO3D'
    cfg["TRAIN"]["LR"] = 5e-5
    cfg["TRAIN"]["BS"] = 24
    cfg["DATA"]["OBJ_SIZE"] = 224
    cfg["TRAIN"]["MAX_EPOCH"] = 600
    cfg["TRAIN"]["STEP_SIZE"] = 75

    cfg["TRAIN"]["FREEZE"] = False
    cfg["TRAIN"]["PRETRAIN"] = True
    cfg["NETWORK"]["MASK"] = "image"
    cfg["NETWORK"]["LOSS"] = "both"
    cfg["DATA"]["FILENAME"] = "checkpoint_co3d"
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', dirpath=os.path.join("./models", cfg["RUN_NAME"]), \
        filename=cfg["DATA"]["FILENAME"])

    logger = TensorBoardLogger("tb_logs", name="SACPose_CO3D")
    
    trainer = pl.Trainer(accelerator="auto", devices=[0,1,2,3,4,5,6,7], strategy="ddp_find_unused_parameters_true", accumulate_grad_batches=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"], sync_batchnorm=True, callbacks=[checkpoint_callback], logger=logger)

    training(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)

