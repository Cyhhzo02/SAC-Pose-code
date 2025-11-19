from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import os
import pytorch_lightning as pl
from modules.sacpose_objaverse import Estimator
from data_loader import Dataset_Loader_Objaverse_stereo_test as Dataset_Loader
from utils import to_cuda

def main(cfg):
    cfg["RUN_NAME"] = 'SACPose_Objaverse'
    cfg["DATA"]["BG"] = True
    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"], map_location=loc_map)
        model.eval()
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    test_dataset = Dataset_Loader(cfg, None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)

    trainer = pl.Trainer()
    model.step_outputs.clear()
    trainer.test(model, dataloaders=test_dataloader)

    pred_err = torch.cat(model.step_outputs)
    gt_dis = torch.cat(model.gt_dis)
    
    pred_err1 = pred_err.cpu().detach().numpy()
    gt_dis1 = gt_dis.cpu().detach().numpy()
    # 合并 pred_err 和 gt_dis
    combined = np.column_stack((pred_err1, gt_dis1))
    # 保存到一个文本文件中
    np.savetxt("./objaverse_combined.txt", combined)
    accuracy_results, num_result, correct_count_num = evaluate_geo_dis_accuracy(pred_err, gt_dis)
    print_accuracy_results(accuracy_results, num_result, correct_count_num)
    pred_acc_30 = 100 * (pred_err < 30).float().mean().item()
    pred_acc_15 = 100 * (pred_err < 15).float().mean().item()
    pred_err = pred_err.mean().item()
    pred_Rs = np.asarray(model.pred_Rs)
    np.savetxt("./objaverse_pred_Rs.txt", pred_Rs)
    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (pred_err, pred_acc_30, pred_acc_15))

    with open("./objaverse_result.txt", 'a') as f:
        f.write("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % (pred_err, pred_acc_30, pred_acc_15))
    f.close()
    
def evaluate_geo_dis_accuracy(geo_dis, gt_rotation_angles, angle_threshold=30):
    accuracy_results = {}
    num_result = {}
    correct_count_num = {}
    angle_bins = range(15, 360, 15)

    for angle in angle_bins:
        indices = (gt_rotation_angles >= angle - 15) & (gt_rotation_angles < angle)
        if torch.sum(indices) == 0:
            accuracy_results[angle] = None
            num_result[angle] = torch.sum(indices).item()
            correct_count_num[angle] = None
            
        else:
            correct_count = torch.sum(geo_dis[indices] < angle_threshold)
            accuracy_results[angle] = (correct_count.item() / torch.sum(indices).item()) * 100
            num_result[angle] = torch.sum(indices).item()
            correct_count_num[angle] = correct_count.item()
    return accuracy_results, num_result, correct_count_num

def print_accuracy_results(accuracy_results, num_result, correct_count_num):
    print("Accuracy Results for Each Geodesic Distance Range:")
    for angle_bin, accuracy in accuracy_results.items():
        if accuracy is not None:
            print(f"Geodesic Distance Range {angle_bin - 15}° to {angle_bin}°: Accuracy = {accuracy:.2f}%, "
                  f"Number of Results = {num_result[angle_bin]}, Correct Count = {correct_count_num[angle_bin]}")
        else:
            print(f"Geodesic Distance Range {angle_bin - 15}° to {angle_bin}°: No data, "
                  f"Number of Results = {num_result[angle_bin]}, Correct Count = {correct_count_num[angle_bin]}")

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
