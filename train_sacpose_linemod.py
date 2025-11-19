from types import SimpleNamespace
import yaml
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from modules.sacpose_objaverse import training
from pytorch_lightning.strategies import DDPStrategy
import warnings
from modules.sacpose_objaverse import training_lm, Estimator

def main(cfg):
    cfg["RUN_NAME"] = 'SACPose_LINEMOD'
    cfg["DATA"]["ACC_THR"] = 15
    cfg["TRAIN"]["BS"] = 24
    cfg["TRAIN"]["LR"] = 5e-5
    cfg["TRAIN"]["MAX_EPOCH"] = 600
    cfg["TRAIN"]["STEP_SIZE"] = 75

    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', dirpath=os.path.join("./models", cfg["RUN_NAME"]), filename="checkpoint_lm")

    logger = TensorBoardLogger("tb_logs", name="SACPose_LINEMOD")
    
    trainer = pl.Trainer(accelerator="auto", devices=[0,1,2,3,4,5,6,7], strategy=DDPStrategy(find_unused_parameters=True), accumulate_grad_batches=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"], sync_batchnorm=True, limit_train_batches=cfg["TRAIN"]["SAMPLE_RATE"],
        callbacks=[checkpoint_callback], logger=logger)

    training_lm(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
