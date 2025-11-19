# SAC-Pose: Structure-Aware Correspondence Learning for Relative Pose Estimation

Official PyTorch implementation for our **CVPR 2025 Highlight** paper:

> **Structure-Aware Correspondence Learning for Relative Pose Estimation**  
> Project Page: <https://cyhhzo02.github.io/SAC-Pose/>  
> Paper: [arXiv 2503.18671](https://arxiv.org/abs/2503.18671)

---

## 1. Overview

Relative pose estimation aims to align two object views without assuming category-specific models. Existing 3D correspondence pipelines still rely on explicit feature matching, which becomes unreliable when overlaps between the two images are small or when large portions are occluded. Inspired by how humans reason about global object structure even with little visible overlap, SAC-Pose introduces two key components:

1. **Structure-aware keypoint extraction.**  
   Guided by a reconstruction loss, the network locates a consistent set of keypoints that describe objects of varying shapes and appearances, even when some regions are invisible.
2. **Structure-aware correspondence estimation.**  
   By modeling both intra-image and inter-image relationships among those keypoints, the network produces structure-aware features that naturally yield reliable 3D–3D correspondences—no explicit feature matching is needed.

Combining these modules allows SAC-Pose to estimate relative pose for unseen objects with significantly improved robustness. On CO3D, Objaverse, and LINEMOD, it surpasses prior SOTA, e.g., achieving a **5.7°** reduction in mean angular error on CO3D.

---

## 2. Installation

```bash
conda create -n sacpose python=3.8 cmake=3.14.0
conda activate sacpose
bash ./install.sh
```

The script installs:

- PyTorch 1.13.0 + CUDA 11.6, torchvision, pytorch3d.
- Auxiliary libs: `fastprogress`, `fvcore`, `iopath`, `lightning`, `opencv-python`, `open3d`, `imutils`, `scipy`, `objaverse`, etc.
- Compiles the CroCo `curope` CUDA extension.

Download CroCo backbone weights:

```bash
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth -P ./croco/
```

> Need a different CUDA/PyTorch combo? Adjust `install.sh` accordingly.

---

## 3. Data Preparation

All dataset processing strictly follows the original [DVMNet instructions via 3DAHV](https://github.com/sailor-z/3DAHV). After generating the processed sets, update `config.yaml`:

```yaml
OBJAVERSE:
  DATA_PATH: ./datasets/Objaverse
  COCO_IMAGE_ROOT: ./datasets/Objaverse/train2017
  COCO_PATH_FILE: ./datasets/Objaverse/img_path.pkl

LINEMOD:
  META_DIR: ./datasets/LINEMOD
  BBOX_FILE: ./datasets/LINEMOD/linemod_bbox.json

CO3D:
  CO3D_DIR: ./datasets/Co3D
  CO3D_ANNOTATION_DIR: ./datasets/Co3D_annotations
```

---

## 4. Configuration

`config.yaml` holds all global settings:

- `DATA`, `TRAIN`, `NETWORK`: dataset sizes, learning rates, losses, mask strategies, etc.
- `SACPOSE`: CroCo checkpoint path, keypoint count, attention depth, Gaussian sigma, heatmap temperature, depth offsets, etc.

Example:

```yaml
SACPOSE:
  BACKBONE_CKPT: ./croco/CroCo_V2_ViTBase_BaseDecoder.pth
  SPATIAL_SIZE: 14
  KEYPOINT_NUM: 48
  KEYPOINT_ATTN_BLOCKS: 4
  KEYPOINT_ATTN_HEADS: 4
  GAUSSIAN_SIGMA: 0.15
  HEATMAP_TEMPERATURE: 0.1
  KPT3D_DEPTH_BASE: 3.0
```

Individual train/test scripts may override a few fields (batch size, LR, etc.) before calling the Lightning trainer.

---

## 5. Pretrained Model

Download the CO3D checkpoint (best single model) from  
<https://drive.google.com/file/d/17XkZ6qn1STLhcbzFOoZl7YJ9ts7JCd0D/view?usp=sharing>  
Place it at `./models/checkpoint_co3d.ckpt`.

---

## 6. Evaluation

```bash
python ./test_co3d_sacpose.py
python ./test_objaverse_sacpose.py
python ./test_linemod_sacpose.py
```

Each script:

- Loads `config.yaml`.
- (Optionally) sets `CUDA_VISIBLE_DEVICES` inside `__main__`.
- Logs errors/accuracies and writes results to `{co3d,objaverse,linemod}_result.txt`.

Reproduced numbers may slightly differ from the paper because evaluation pairs follow RelPose++’s random sampling.

---

## 7. Training

```bash
python ./train_sacpose_co3d.py
python ./train_sacpose_objaverse.py
python ./train_sacpose_linemod.py
```

Scripts configure `RUN_NAME`, optimizer hyper-parameters, and Lightning Trainer options before calling the dataset-specific `training` function. Modify devices, strategy, mixed precision, etc., according to your hardware.

---

## 8. Repository Layout

```
SAC-Pose-code/
├── config.yaml
├── install.sh
├── croco/                     # CroCo backbone (from original release)
├── modules/
│   ├── sacpose_modules.py     # SACPoseModel
│   ├── sacpose_co3d.py        # Lightning module for Co3D
│   └── sacpose_objaverse.py   # Lightning module for Objaverse / LINEMOD
├── data_loader.py
├── data_loader_co3d.py
├── train_sacpose_*.py
├── test_sacpose_*.py
└── utils*.py
```

---

## 9. Citation

```bibtex
@inproceedings{chen2025structure,
  title={Structure-Aware Correspondence Learning for Relative Pose Estimation},
  author={Chen, Yihan and Yang, Wenfei and Ren, Huan and Zhang, Shifeng and Zhang, Tianzhu and Wu, Feng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={11611--11621},
  year={2025}
}
```

---

## 10. Acknowledgements

Our implementation is heavily based on three excellent open-source projects:

- [DVMNet](https://github.com/sailor-z/DVMNet/tree/main) – end-to-end relative pose framework and dataset split/preprocessing code.
- [3DAHV](https://github.com/sailor-z/3DAHV) – detailed data preparation scripts for Co3D, Objaverse, and LINEMOD.
- [CroCo](https://github.com/naver/croco) – pre-trained transformer encoders and reconstruction utilities.

We sincerely thank the original authors for sharing their code and datasets.
