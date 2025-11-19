import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
import os
import time
from fastprogress import progress_bar
import pytorch_lightning as pl
from modules.sacpose_objaverse import Estimator
from data_loader import Dataset_Loader_LINEMOD_stereo as Dataset_Loader
from utils import to_cuda, get_calibration_matrix_K_from_blender
from utils_3dahv import visualization
from LoFTR.src.loftr import LoFTR, default_cfg

CATEGORY_LM = ["CAT", "BENCHVISE", "CAM", "DRILLER", "DUCK"]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def test_category(cfg, model, clsID):
    K = np.array(cfg["LINEMOD"]["INTERNAL_K"]).reshape(3, 3)
    test_dataset = Dataset_Loader(cfg, clsID)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)
    pbar = progress_bar(test_dataloader, leave=False)
    pred_Rs, pred_errs = [], []
    for idx, data in enumerate(pbar):
        data = to_cuda(data)
        mask_src, mask_tgt = data["src_mask"], data["ref_mask"]
        img_src, img_tgt = data["src_img"], data["ref_img"]
        R_src, R_tgt = data["src_R"], data["ref_R"]
        T_src, T_tgt = data["src_T"], data["ref_T"]
        crop_params_src, crop_params_tgt = data["src_crop_params"], data["ref_crop_params"]

        if torch.any(mask_src.flatten(1).sum(dim=-1) < cfg["DATA"]["SIZE_THR"]) or torch.any(mask_tgt.flatten(1).sum(dim=-1) < cfg["DATA"]["SIZE_THR"]):
            print("Skip bad case")
            continue
        img_recon_src, img_recon_tgt, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source = model(img_src, img_tgt)
        gt_delta_R = torch.bmm(R_tgt, torch.inverse(R_src))
        sim = (torch.sum(pred_delta_R.view(-1, 9) * gt_delta_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi
        pred_errs.append(geo_dis)
        pbar.comment = "Error: %.2f" % (geo_dis.mean().item())
        pred_Rs.append(pred_delta_R.cpu().detach().numpy().reshape(-1))
    pred_err = torch.cat(pred_errs)
    pred_acc_30 = 100 * (pred_err < 30).float().mean().item()
    pred_acc_15 = 100 * (pred_err < 15).float().mean().item()
    pred_err = pred_err.mean().item()
    pred_Rs = np.asarray(pred_Rs)
    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (pred_err, pred_acc_30, pred_acc_15))

    return pred_err, pred_acc_30, pred_acc_15

def main(cfg):
    cfg["RUN_NAME"] = 'SACPose_Linemod'
    cfg["DATA"]["BG"] = True
    cfg["TRAIN"]["FREEZE"] = False
    cfg["TRAIN"]["PRETRAIN"] = True
    cfg["NETWORK"]["MASK"] = "both"
    cfg["NETWORK"]["LOSS"] = "both"

    filename = "checkpoint_linemod.ckpt"
    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], filename)
    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"], map_location=loc_map)
        model = model.model.eval()
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    errs, pred_accs_30, pred_accs_15 = [], [], []

    for obj in CATEGORY_LM:
        with torch.no_grad():
            err, pred_acc_30, pred_acc_15 = test_category(cfg, model, cfg["LINEMOD"][obj])
        with open('./linemod_result.txt', 'a') as f:
            f.write("%s -- err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % (obj, err, pred_acc_30, pred_acc_15))
        f.close()

        errs.append(err)
        pred_accs_30.append(pred_acc_30)
        pred_accs_15.append(pred_acc_15)

    err = np.asarray(errs).mean()
    pred_acc_30 = np.asarray(pred_accs_30).mean()
    pred_acc_15 = np.asarray(pred_accs_15).mean()

    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (err, pred_acc_30, pred_acc_15))

    with open('./linemod_result.txt', 'a') as f:
        f.write("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % (err, pred_acc_30, pred_acc_15))
    f.close()

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
