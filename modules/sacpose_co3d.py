import argparse, logging, copy, sys
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
                                
from torchvision.utils import make_grid
import pytorch_lightning as pl
import cv2
from einops import rearrange
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations
from data_loader_co3d import Co3dDataset as Dataset_Loader
from modules.sacpose_modules import MaskedMSE, SacPoseModel
from modules.utils_vgg_loss import VGGLoss
from torchvision import transforms
from utils import *
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)

def training(cfg, trainer):
    model = Estimator(cfg, img_size=cfg["DATA"]["OBJ_SIZE"], num_rota=cfg["DATA"]["NUM_ROTA"], freeze=cfg["TRAIN"]["FREEZE"])

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["DATA"]["OBJ_SIZE"], antialias=True),                                 
        ]
    )
    train_dataset = Dataset_Loader(
        cfg=cfg,
        category=["all"],
        split="train",
        random_aug=True,
        eval_time=False,
        num_images=2,
        normalize_cameras=False,
        transform=trans,
        img_size=cfg["DATA"]["OBJ_SIZE"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)                    
    ckpt_path = os.path.join("models", cfg["RUN_NAME"], cfg["DATA"]["FILENAME"]+'.ckpt')
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)

class Estimator(pl.LightningModule):
    def __init__(self, cfg, img_size=224, num_rota=50000, freeze=True):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.num_rota = num_rota
        self.freeze = freeze
        self.alpha = 0.3
        self.transition_start_epoch = cfg.get("TRAIN", {}).get("TRANSITION_START_EPOCH", 0)
        self.transition_end_epoch = cfg.get("TRAIN", {}).get("TRANSITION_END_EPOCH", 600)                                                                         
        sacpose_cfg = cfg.get("SACPOSE", {})
        self.model = SacPoseModel(cfg=cfg, transport="cosine", mask=cfg["NETWORK"]["MASK"])
        patch_size = sacpose_cfg.get("PATCH_SIZE", 16)
        self.criterion = MaskedMSE(norm_pix_loss=True, masked=True, patch_size=patch_size)
        self.percep_loss = VGGLoss()
        self.mse_loss = nn.MSELoss()        
        self.log_interval = cfg.get("LOG_INTERVAL", 500) 

    def training_step(self, batch, batch_idx):
        img_src = batch["image"][:, 0]
        img_tgt = batch["image"][:, 1]                 
        mask_src = batch["mask"][:, 0]
        mask_tgt = batch["mask"][:, 1]                 
        R_src = batch["R_cv"][:, 0]
        R_tgt = batch["R_cv"][:, 1] 
        T_src = batch['T_cv'][:, 0]
        T_tgt = batch['T_cv'][:, 1]
        h, w = img_src.shape[-2:]
        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))
        gt_delta_R_inverse = torch.bmm(R_src, torch.inverse(R_tgt))
        bs = img_src.shape[0]
                          
        outputs = self.model(img_src, img_tgt, gt_delta_R, gt_delta_R_inverse, 
                           current_epoch=self.current_epoch, 
                           total_epochs=self.transition_end_epoch)
        
        img_recon_src, img_recon_tgt, img_recon_src_r, img_recon_tgt_r, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source = outputs

        num_kpts = kpt_loc_src.shape[1]
        kpt_loc_src_pixel = (kpt_loc_src * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()                             
        kpt_loc_tgt_pixel = (kpt_loc_tgt * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()
                
        batch_indices = torch.arange(bs).view(-1, 1).expand(-1, num_kpts).to(kpt_loc_src.device)         
        kpt_mask_src = mask_src.squeeze(1)[batch_indices, kpt_loc_src_pixel[..., 0], kpt_loc_src_pixel[..., 1]]
        kpt_mask_tgt = mask_tgt.squeeze(1)[batch_indices, kpt_loc_tgt_pixel[..., 0], kpt_loc_tgt_pixel[..., 1]]                                                                                                                                    
        kpt3d_dir_target_gt = torch.bmm(gt_delta_R_inverse, kpt3d_anchor_target.transpose(1, 2)).transpose(1, 2)           
        kpt3d_dir_source_gt = torch.bmm(gt_delta_R, kpt3d_anchor_source.transpose(1, 2)).transpose(1, 2)           
        loss_kpt_in_mask = torch.norm((1 - kpt_mask_src), p=2, dim=1).mean() + torch.norm((1 - kpt_mask_tgt), p=2, dim=1).mean()
        loss_mask_src = F.binary_cross_entropy(img_mask_src, mask_src)
        loss_mask_tgt = F.binary_cross_entropy(img_mask_tgt, mask_tgt)
        loss_mask = loss_mask_src + loss_mask_tgt

        loss_mse_recon_src = 5 * self.mse_loss(img_recon_src * mask_src, img_src * mask_src)
        loss_percep_recon_src = self.percep_loss(img_recon_src * mask_src, img_src * mask_src)
        loss_recon_src = loss_mse_recon_src + loss_percep_recon_src
        loss_mse_recon_tgt = 5 * self.mse_loss(img_recon_tgt * mask_tgt, img_tgt * mask_tgt) 
        loss_percep_recon_tgt = self.percep_loss(img_recon_tgt * mask_tgt, img_tgt * mask_tgt)
        loss_recon_tgt = loss_mse_recon_tgt + loss_percep_recon_tgt

        loss_mse_recon_src_r = 5 * self.mse_loss(img_recon_src_r * mask_src, img_src * mask_src)
        loss_percep_recon_src_r = self.percep_loss(img_recon_src_r * mask_src, img_src * mask_src)
        loss_recon_src_r = loss_mse_recon_src_r + loss_percep_recon_src_r
        loss_mse_recon_tgt_r = 5 * self.mse_loss(img_recon_tgt_r * mask_tgt, img_tgt * mask_tgt)
        loss_percep_recon_tgt_r = self.percep_loss(img_recon_tgt_r * mask_tgt, img_tgt * mask_tgt)
        loss_recon_tgt_r = loss_mse_recon_tgt_r + loss_percep_recon_tgt_r                                                                                                                                    
        
        gt_rota_6d = matrix_to_rotation_6d(gt_delta_R)
        pred_rota_6d = matrix_to_rotation_6d(pred_delta_R)
        pred_rota_6d_inverse = matrix_to_rotation_6d(pred_delta_R_inverse)
        gt_rota_6d_inverse = matrix_to_rotation_6d(gt_delta_R_inverse)
        loss_rota = torch.norm(pred_rota_6d - gt_rota_6d.detach(), p=1, dim=-1).mean() + torch.norm(pred_rota_6d_inverse - gt_rota_6d_inverse.detach(), p=1, dim=-1).mean()
                                             
        e_norm_tgt = (torch.norm(kpt3d_dir_target - kpt3d_dir_target_gt.detach(), p=2, dim=-1) + torch.norm(kpt3d_dir_target - kpt3d_dir_target_gt.detach(), p=2, dim=-1)) / 2
        e_norm_src = (torch.norm(kpt3d_dir_source - kpt3d_dir_source_gt.detach(), p=2, dim=-1) + torch.norm(kpt3d_dir_source - kpt3d_dir_source_gt.detach(), p=2, dim=-1)) / 2
        
        loss_kpt3d = ((torch.arccosh(1 + e_norm_tgt) * kpt3d_weight_target - self.alpha * torch.log(kpt3d_weight_target))).mean() +\
                    ((torch.arccosh(1 + e_norm_src) * kpt3d_weight_source - self.alpha * torch.log(kpt3d_weight_source))).mean()
        if self.cfg["NETWORK"]["LOSS"] == "rota":
            loss = loss_rota
        elif self.cfg["NETWORK"]["LOSS"] == "both":                                                                             
            loss = loss_recon_src * 1 + loss_recon_tgt * 1 + loss_mask * 10 + loss_rota * 2 + loss_kpt3d*1 + loss_kpt_in_mask * 0 + loss_recon_src_r * 1 + loss_recon_tgt_r * 1                                                                                                               
        else:
            raise RuntimeError("Unsupported loss function")

        current_lr = self.optimizers().param_groups[0]['lr']                                                                
        self.log_dict({'learning_rate': current_lr, 
                    "train_loss": loss.item(), 
                    "loss_recon_src": loss_recon_src.item(), 
                    "loss_recon_tgt": loss_recon_tgt.item(), 
                    "loss_recon_src_mse": loss_mse_recon_src.item(),
                    "loss_recon_tgt_mse": loss_mse_recon_tgt.item(),
                    "loss_recon_src_percep": loss_percep_recon_src.item(),
                    "loss_recon_tgt_percep": loss_percep_recon_tgt.item(),
                    "loss_kpt3d": loss_kpt3d.item(),                                               
                    "loss_mask": loss_mask.item(),
                    "loss_rota": loss_rota.item() * 1,
                    'score': kpt3d_weight_target.mean(),
                    'loss_kpt_in_mask': loss_kpt_in_mask.item(),
                    'loss_recon_src_r': loss_recon_src_r.item(),
                    'loss_recon_tgt_r': loss_recon_tgt_r.item(),
                    },
                   on_step=True, 
                   prog_bar=True, 
                   logger=True, 
                   sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg["TRAIN"]["STEP_SIZE"], gamma=0.5)

        return [optimizer], [scheduler]
    
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
    
    def norm_dist(self, x1, x2, x3, x4):
        '''
        input: x_i: bs, k,  3  -> x: bs, 4k,  3
        output: bs ,1, 1
        '''
        x = torch.concat([x1, x2, x3, x4], dim=1)
        norm_x = torch.norm(x, p=2, dim=2)            
        norm_x = norm_x.mean(dim=1)     
        
        return norm_x.unsqueeze(-1).unsqueeze(-1)
    
    def find_nearest_nonzero(self, kpt_loc_pixel, pointmap, pointmap_mask):
        batch_size, num_kpts, _ = kpt_loc_pixel.shape
        height, width = pointmap.shape[1:3]    
        non_zero_points = torch.nonzero(~pointmap_mask)
        nearest_pixel = kpt_loc_pixel.clone()

        for b in range(batch_size):
            non_zero_points_batch = non_zero_points[non_zero_points[:, 0] == b][:, 1:]
            kpt_loc_pixel_batch = kpt_loc_pixel[b]

            if non_zero_points_batch.size(0) == 0:
                continue              
            distances = torch.cdist(kpt_loc_pixel_batch.float().unsqueeze(0), non_zero_points_batch.float().unsqueeze(0)).squeeze(0)              
            nearest_indices = torch.argmin(distances, dim=-1)
            nearest_pixel[b] = non_zero_points_batch[nearest_indices]

        return nearest_pixel

    def find_valid_pos(self, kpt_loc_pixel, pointmap):
        '''
        pointmap: bs 224 224 3
        kpt_loc_pixel: bs 16 2
        '''
        bs, num_kpts = kpt_loc_pixel.shape[:2]
                
        batch_indices = torch.arange(bs).view(-1, 1).expand(-1, num_kpts).to(kpt_loc_pixel.device)
                  
        kpt_3d_coord_gt = pointmap[batch_indices, kpt_loc_pixel[..., 0], kpt_loc_pixel[..., 1]]
        kpt_mask = kpt_3d_coord_gt.sum(dim=-1) != 0
        
                               
        invalid_mask = ~kpt_mask
        pointmap_mask = (pointmap == torch.tensor([0, 0, 0], device=pointmap.device)).all(dim=-1)
        if invalid_mask.any():
            new_kpt_loc_pixel = self.find_nearest_nonzero(kpt_loc_pixel, pointmap, pointmap_mask)
            kpt_loc_pixel[invalid_mask] = new_kpt_loc_pixel[invalid_mask]
            kpt_3d_coord_gt = pointmap[batch_indices, kpt_loc_pixel[..., 0], kpt_loc_pixel[..., 1]]
            kpt_mask = kpt_3d_coord_gt.sum(dim=-1) != 0
        
        return kpt_3d_coord_gt
