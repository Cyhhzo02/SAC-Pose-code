import argparse, logging, copy, sys, math
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms                               
import pytorch_lightning as pl
import numpy as np
import cv2
from utils import *
from data_loader import Dataset_Loader_Objaverse_stereo as Dataset_Loader
from data_loader import Dataset_Loader_Objaverse_stereo_test as Dataset_Loader_Test
from data_loader import Dataset_Loader_LINEMOD_stereo_train as Dataset_Loader_LM
from modules.sacpose_modules import SacPoseModel
from modules.utils_vgg_loss import VGGLossObjaverse
from torchvision.utils import make_grid
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)
class Estimator(pl.LightningModule):
    def __init__(self, cfg=None, img_size=224):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.alpha = 0.3
        self.model = SacPoseModel(cfg=cfg, transport="cosine", mask=cfg["NETWORK"]["MASK"])
        self.percep_loss = VGGLossObjaverse()
        self.mse_loss = nn.MSELoss()
        self.log_interval = cfg.get("LOG_INTERVAL", 500) 
        self.step_outputs = []
        self.gt_dis = []
        self.pred_Rs = []

    def training_step(self, batch, batch_idx):                                       
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]                       
        if self.cfg["DATA"]["BG"] is False:
            img_src = img_src * mask_src
            img_tgt = img_tgt * mask_tgt
        with torch.no_grad():
            gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))
            gt_delta_R_inverse = torch.bmm(R_src, torch.inverse(R_tgt))
        h, w = img_src.shape[-2:]                                                    
        bs = img_src.shape[0]
        img_recon_src, img_recon_tgt, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source = self.model(img_src, img_tgt)
        num_kpts = kpt_loc_src.shape[1]
        kpt_loc_src_pixel = (kpt_loc_src * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()                             
        kpt_loc_tgt_pixel = (kpt_loc_tgt * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()        
        batch_indices = torch.arange(bs).view(-1, 1).expand(-1, num_kpts).to(kpt_loc_src.device)         

        kpt_mask_src = mask_src.squeeze(1)[batch_indices, kpt_loc_src_pixel[..., 0], kpt_loc_src_pixel[..., 1]]
        kpt_mask_tgt = mask_tgt.squeeze(1)[batch_indices, kpt_loc_tgt_pixel[..., 0], kpt_loc_tgt_pixel[..., 1]]                
        
        kpt3d_dir_target_gt = torch.bmm(gt_delta_R, kpt3d_anchor_target.transpose(1, 2)).transpose(1, 2)           
        kpt3d_dir_source_gt = torch.bmm(gt_delta_R_inverse, kpt3d_anchor_source.transpose(1, 2)).transpose(1, 2)           

        loss_kpt_in_mask = torch.norm((1 - kpt_mask_src), p=2, dim=1).mean() + torch.norm((1 - kpt_mask_tgt), p=2, dim=1).mean()
                                                
        loss_mask_src = F.binary_cross_entropy(img_mask_src, mask_src, reduction='none').mean(dim=[1, 2, 3])
        loss_mask_tgt = F.binary_cross_entropy(img_mask_tgt, mask_tgt, reduction='none').mean(dim=[1, 2, 3])
        loss_mask = loss_mask_src + loss_mask_tgt                 
            
        loss_mse_recon_src = 5 * F.mse_loss(
            img_recon_src * mask_src, img_src * mask_src, reduction='none'
        ).mean(dim=[1, 2, 3])
        loss_mse_recon_tgt = 5 * F.mse_loss(
            img_recon_tgt * mask_tgt, img_tgt * mask_tgt, reduction='none'
        ).mean(dim=[1, 2, 3])         
        loss_percep_recon_src = self.percep_loss(
            img_recon_src * mask_src, img_src * mask_src
        )
        loss_percep_recon_tgt = self.percep_loss(
            img_recon_tgt * mask_tgt, img_tgt * mask_tgt
        )           
        loss_recon_src = loss_mse_recon_src + loss_percep_recon_src                 
        loss_recon_tgt = loss_mse_recon_tgt + loss_percep_recon_tgt                                                                                                                                                  
        gt_rota_6d = matrix_to_rotation_6d(gt_delta_R)
        pred_rota_6d = matrix_to_rotation_6d(pred_delta_R)
        pred_rota_6d_inverse = matrix_to_rotation_6d(pred_delta_R_inverse)
        gt_rota_6d_inverse = matrix_to_rotation_6d(gt_delta_R_inverse)
        loss_rota = torch.norm(pred_rota_6d - gt_rota_6d.detach(), p=1, dim=-1) + torch.norm(pred_rota_6d_inverse - gt_rota_6d_inverse.detach(), p=1, dim=-1)
        e_norm_tgt = (torch.norm(kpt3d_dir_target - kpt3d_dir_target_gt.detach(), p=2, dim=-1) + torch.norm(kpt3d_dir_target.detach() - kpt3d_dir_target_gt, p=2, dim=-1)) / 2
        e_norm_src = (torch.norm(kpt3d_dir_source - kpt3d_dir_source_gt.detach(), p=2, dim=-1) + torch.norm(kpt3d_dir_source.detach() - kpt3d_dir_source_gt, p=2, dim=-1)) / 2
        loss_kpt3d = (e_norm_src * kpt3d_weight_source - self.alpha * torch.log(kpt3d_weight_source)).mean(1) +\
                    (e_norm_tgt * kpt3d_weight_target - self.alpha * torch.log(kpt3d_weight_target)).mean(1)
        valid = (mask_src.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"]) * (mask_tgt.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"])
        if "dis_init" in batch.keys():
            dis_init = batch["dis_init"]
            valid = valid * (dis_init < self.cfg["DATA"]["VIEW_THR"]).float()

        loss_recon_src = loss_recon_src * valid
        loss_recon_src = loss_recon_src.sum() / valid.sum().clamp(min=1e-8)

        loss_recon_tgt = loss_recon_tgt * valid
        loss_recon_tgt = loss_recon_tgt.sum() / valid.sum().clamp(min=1e-8)

        loss_mask = loss_mask * valid
        loss_mask = loss_mask.sum() / valid.sum().clamp(min=1e-8)
        
        loss_rota = loss_rota * valid
        loss_rota = loss_rota.sum() / valid.sum().clamp(min=1e-8)
        
        loss_kpt3d = loss_kpt3d * valid
        loss_kpt3d = loss_kpt3d.sum() / valid.sum().clamp(min=1e-8)

        if self.cfg["NETWORK"]["LOSS"] == "rota":
            loss = loss_rota
        elif self.cfg["NETWORK"]["LOSS"] == "both":                                                                              
            loss = loss_recon_src * 1 + loss_recon_tgt * 1 + loss_mask * 10 + loss_rota * 2 + loss_kpt3d*1 + loss_kpt_in_mask * 0                                                                                                    
        else:
            raise RuntimeError("Unsupported loss function")
        current_lr = self.optimizers().param_groups[0]['lr']                                                                 
        self.log_dict({'learning_rate': current_lr, 
                    "train_loss": loss.item(), 
                    "loss_recon_src": loss_recon_src.item(), 
                    "loss_recon_tgt": loss_recon_tgt.item(),                                                                             
                    "loss_kpt3d": loss_kpt3d.item(),                                                      
                    "loss_mask": loss_mask.item(),
                    "loss_rota": loss_rota.item() * 1,
                    'score': kpt3d_weight_target.mean(),                                         
                    }, 
                   on_step=True, 
                   prog_bar=True, 
                   logger=True, 
                   sync_dist=True)
                                                
        return loss
    
    def visualize_and_log(self, images, keypoints, tag):
        """
        Visualize images with keypoints and log them using the provided logger.

        Args:
            images (tensor): Input images of shape (B, C, H, W).
            keypoints (tensor): Keypoints of shape (B, N, 2).
            kpt_conf (tensor): Keypoint confidences of shape (B, N).
            tag (str): Tag for logging.
        """                    
        nrow = int(np.sqrt(images.size(0)))
        grid_img = make_grid(images, nrow=nrow, padding=2)
        np_img = grid_img.permute(1, 2, 0).cpu().detach().numpy()                       
        fig, ax = plt.subplots(figsize=(20, 20))                                
        ax.imshow(np_img)
        img_height, img_width = images.size(2), images.size(3)
        padding = 2
        for idx, kp in enumerate(keypoints):
            kp = kp.detach().cpu().numpy()                                              
            row = idx // nrow
            col = idx % nrow                                                      
            y_offset = row * (img_height + padding)
            x_offset = col * (img_width + padding)
            for y, x in kp:
                y_pos = y_offset + y * img_height
                x_pos = x_offset + x * img_width
                ax.scatter(x_pos, y_pos, s=10, c='red', marker='o', alpha=0.6)                        
        self.logger.experiment.add_figure(tag, fig, self.global_step)
        plt.close(fig)
    
    def validation_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]
        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))                                                                                                                                                                                          
        img_recon_src, img_recon_tgt, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source = self.model(img_src, img_tgt)
                   
        sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        pred_acc_15 = (geo_dis <= 15).float().mean()
        pred_acc_30 = (geo_dis <= 30).float().mean()

        self.log("val_geo_dis", geo_dis.float().mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_15", pred_acc_15.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_30", pred_acc_30.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.step_outputs.append(geo_dis)

    def on_validation_epoch_end(self):
        geo_dis = torch.cat(self.step_outputs)
        pred_acc_15 = 100 * (geo_dis <= 15).float().mean()
        pred_acc_30 = 100 * (geo_dis <= 30).float().mean()
        geo_dis = geo_dis.float().mean()

        self.step_outputs.clear()

    def test_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        if torch.any(mask_src.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]) or torch.any(mask_tgt.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]):
            print("Skip bad case")
            return 0

        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))
        img_recon_src, img_recon_tgt, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source = self.model(img_src, img_tgt)
        sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        trace = gt_delta_R[..., 0, 0] + gt_delta_R[..., 1, 1] + gt_delta_R[..., 2, 2]
        gt = (trace.clamp(-1, 3) - 1) / 2
                                                                               
        geo_dis_gt = torch.arccos(gt) * 180. / np.pi
        geo_dis = torch.arccos(sim) * 180. / np.pi

        self.gt_dis.append(geo_dis_gt)
        self.step_outputs.append(geo_dis)
        self.pred_Rs.append(pred_delta_R.cpu().detach().numpy().reshape(-1))

        self.log("test_error", geo_dis.mean().item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg["TRAIN"]["STEP_SIZE"], gamma=0.5)
        return [optimizer], [scheduler]

def training(cfg, trainer):
    val_dataset = Dataset_Loader_Test(cfg, None)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)
    
    train_dataset = Dataset_Loader(cfg, "train", None)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)
    
    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    ckpt_path = os.path.join("models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader, val_dataloader)

def training_lm(cfg, trainer):
    CATEGORY = ["APE", "CAN", "EGGBOX", "GLUE", "HOLEPUNCHER", "IRON", "LAMP", "PHONE"]
    clsIDs = [cfg["LINEMOD"][cat] for cat in CATEGORY]

    train_dataset = Dataset_Loader_LM(cfg, clsIDs)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        loc_map = {"cuda:3": "cuda:5", "cuda:5": "cuda:5"}
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"], map_location=loc_map)
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    filename = "checkpoint_lm.ckpt"

    ckpt_path = os.path.join("models", cfg["RUN_NAME"], filename)
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)
