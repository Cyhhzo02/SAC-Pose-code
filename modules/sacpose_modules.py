import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import patchify, unpatchify, log_optimal_transport
from einops import rearrange
from functools import partial
from modules.modules_utils import AttnLayer, ResNet_Decoder
from torchvision import transforms
import math
torch.backends.cuda.matmul.allow_tf32 = True                                        
sys.path.append("./croco/")
from models.croco import CroCoNet
from models.blocks import DecoderBlock, DecoderBlock_Monocular, DecoderBlock_query_embedding_pos, DecoderBlockNoSelf, DecoderBlockNoCross
from models.pos_embed import RoPE2D

class PositionGetter(object):
    """ return positions of patches """
    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x)            
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos
class MaskedMSE(torch.nn.Module):
    def __init__(self, norm_pix_loss=False, masked=True, patch_size=16):
        """
            norm_pix_loss: normalize each patch by their pixel mean and variance
            masked: compute loss over the masked patches only
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked
        self.patch_size = patch_size

    def forward(self, pred, target, mask):
        pred = unpatchify(pred, patch_size=self.patch_size, channels=4)
        pred_img = pred[:, :3]
        pred_mask = pred[:, 3]
        if self.norm_pix_loss:
            with torch.no_grad():
                target = patchify(target, patch_size=self.patch_size)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
                target = unpatchify(target, patch_size=self.patch_size, channels=3)
        loss_img = (pred_img - target.detach()) ** 2
        loss_img = loss_img.sum(dim=1).flatten(1)
        mask = mask.squeeze(1).flatten(1)
        if self.masked:
            loss_img = (loss_img * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)                                 
        else:
            loss_img = loss_img.mean(dim=-1)       
        loss_mask = F.binary_cross_entropy_with_logits(pred_mask.flatten(1), mask.detach(), reduction='none').mean(dim=-1)

        return loss_img + loss_mask

class SacPoseModel(nn.Module):
    def __init__(self, cfg=None, transport=None, mask="both"):
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("SACPOSE", {})
        data_cfg = cfg.get("DATA", {})
        self.spatial_size = model_cfg.get("SPATIAL_SIZE", 14)
        self.patch_size = model_cfg.get("PATCH_SIZE", 16)
        self.in_channels = model_cfg.get("IN_CHANNELS", 768)
        self.embed_channels = model_cfg.get("EMBED_CHANNELS", self.in_channels)
        self.decode_channels = model_cfg.get("DECODE_CHANNELS", self.embed_channels)
        self.keypoint_num = model_cfg.get("KEYPOINT_NUM", 48)
        self.feat_dim = self.embed_channels
        self.n_heads = model_cfg.get("N_HEADS", 8)
        self.cross_depth = model_cfg.get("CROSS_DEPTH", 3)
        self.de_depth = model_cfg.get("DECODER_DEPTH", 1)
        self.keypoint_cross_depth = model_cfg.get("KEYPOINT_CROSS_DEPTH", 4)
        self.keypoint_attn_blocks = model_cfg.get("KEYPOINT_ATTN_BLOCKS", 4)
        self.keypoint_attn_heads = model_cfg.get("KEYPOINT_ATTN_HEADS", 4)
        self.num_pe_bases = model_cfg.get("NUM_PE_BASES", 8)
        self.kpt3d_depth_base = model_cfg.get("KPT3D_DEPTH_BASE", 3.0)
        self.gaussian_sigma = model_cfg.get("GAUSSIAN_SIGMA", 0.15)
        self.heatmap_temperature = max(model_cfg.get("HEATMAP_TEMPERATURE", 0.1), 1e-6)
        backbone_ckpt = model_cfg.get("BACKBONE_CKPT", "./croco/CroCo_V2_ViTBase_BaseDecoder.pth")
        backbone_device = model_cfg.get("BACKBONE_DEVICE", "cpu")
        self.volume_channels = self.decode_channels // self.spatial_size - 1
        ckpt = torch.load(backbone_ckpt, map_location=backbone_device)
        self.backbone = CroCoNet(**ckpt.get('croco_kwargs',{}))
        self.backbone.load_state_dict(ckpt['model'], strict=True)
        self.keypoint_query = nn.Embedding(self.keypoint_num, self.feat_dim)
        self.keypoint_pos_embedding = nn.Embedding(self.keypoint_num, self.feat_dim)
        self.keypoint_detector = AttnLayer(None, self.keypoint_attn_blocks, self.embed_channels, self.keypoint_attn_heads, self.embed_channels)
        self.keypoint_cross_blocks = nn.ModuleList([
            DecoderBlock(self.embed_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for _ in range(self.keypoint_cross_depth)])
        self.position_getter = PositionGetter()
        self.cross_att_blocks_src = nn.ModuleList([
            DecoderBlock(self.in_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for _ in range(self.cross_depth)])
        self.encoder_embed = nn.Linear(self.in_channels, self.decode_channels)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock_Monocular(self.decode_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for _ in range(self.de_depth)])
        self.prediction_head = nn.Sequential(
             nn.LayerNorm(self.decode_channels),
             nn.Linear(self.decode_channels, self.patch_size**2, bias=True),
             nn.Sigmoid()
        )
        self.register_buffer("embed_crop_src", (2 ** torch.arange(self.num_pe_bases)).reshape(1, 1, -1))
        self.register_buffer("embed_crop_tgt", (2 ** torch.arange(self.num_pe_bases)).reshape(1, 1, -1))
        self.svd_head = SVDHead(transport, mask)
        self.reconstructer = ResNet_Decoder(self.decode_channels)
        self.rotation_encoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_channels)
        )
        self.fusion_block = nn.Sequential(
            nn.Conv2d(self.embed_channels * 2, self.embed_channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.embed_channels, self.embed_channels, 1)
        )
        self.kpt3d_head = nn.Sequential(
            nn.Linear(self.embed_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)   
        )
        self.transform = transforms.Normalize(
            data_cfg.get("PIXEL_MEAN", [0.485, 0.456, 0.406]),
            data_cfg.get("PIXEL_STD", [0.229, 0.224, 0.225])
        )
    def positional_encoding(self, x, embedding):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        embed = (x[..., None] * embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def encoding(self, img_src, img_tgt):
        bs = img_src.shape[0]
        img_src_transformed = self.transform(img_src)
        img_tgt_transformed = self.transform(img_tgt)
        feat_src = self.backbone._encode_image(img_src_transformed, do_mask=False, return_all_blocks=False)[0]
        feat_tgt = self.backbone._encode_image(img_tgt_transformed, do_mask=False, return_all_blocks=False)[0]
        pos_src = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_src.device)
        pos_tgt = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_tgt.device)
        for idx, blk_src in enumerate(self.cross_att_blocks_src):
            if idx == 0:                                                      
                (feat_src_, _), (feat_tgt_, _) = blk_src(feat_src, feat_tgt, pos_src, pos_tgt), blk_src(feat_tgt, feat_src, pos_tgt, pos_src)
            else:                                       
                (feat_src_, _), (feat_tgt_, _) = blk_src(feat_src_, feat_tgt_, pos_src, pos_tgt), blk_src(feat_tgt_, feat_src_, pos_tgt, pos_src)
        feat_src = self.encoder_embed(feat_src_).transpose(1, 2)                            
        feat_tgt = self.encoder_embed(feat_tgt_).transpose(1, 2)

        return feat_src, feat_tgt
    
    def mask_predictor(self, feat):
        '''
        feat: B C (HW)
        '''
        h = self.spatial_size
        w = self.spatial_size
        b = feat.size(0)                                                     
        feat = feat.transpose(1, 2)
        pos = self.position_getter(b, h, w, feat.device)
        for blk in self.dec_blocks:
            feat = blk(feat, pos)
        pred_mask = self.prediction_head(feat)               

        return pred_mask

    def get_kpt(self, feat_src, feat_tgt):
        bs = feat_src.size(0)
        c = feat_src.size(1)
        h = self.spatial_size
        w = self.spatial_size
        
        kpt_query = self.keypoint_query.weight.squeeze(0).repeat(feat_src.size(0), 1, 1)
        kpt_pos_emb = self.keypoint_pos_embedding.weight.squeeze(0).repeat(feat_src.size(0), 1, 1)

        kpt_feat_src, attn_src = self.keypoint_detector(kpt_query + kpt_pos_emb, feat_src)
        kpt_feat_tgt, attn_tgt = self.keypoint_detector(kpt_query + kpt_pos_emb, feat_tgt)          

        kpt_feat_src_norm = torch.norm(kpt_feat_src, p=2, dim=2, keepdim=True) 
        feat_src_norm = torch.norm(feat_src, p=2, dim=1, keepdim=True) 
        heatmap_src = torch.bmm(kpt_feat_src, feat_src) / (kpt_feat_src_norm * feat_src_norm + 1e-7)
        heatmap_src = F.softmax(heatmap_src / self.heatmap_temperature, dim=2)
        
        kpt_feat_tgt_norm = torch.norm(kpt_feat_src, p=2, dim=2, keepdim=True) 
        feat_tgt_norm = torch.norm(feat_tgt, p=2, dim=1, keepdim=True) 
        heatmap_tgt = torch.bmm(kpt_feat_tgt, feat_tgt) / (kpt_feat_tgt_norm * feat_tgt_norm + 1e-7)
        heatmap_tgt = F.softmax(heatmap_tgt / self.heatmap_temperature, dim=2)
        
        kpt_feat_src = torch.matmul(heatmap_src, feat_src.view(bs, c, -1).permute(0, 2, 1))
        kpt_feat_tgt = torch.matmul(heatmap_tgt, feat_tgt.view(bs, c, -1).permute(0, 2, 1))
        
        heatmap_src = heatmap_src.reshape(bs, self.keypoint_num, h, w)
        heatmap_tgt = heatmap_tgt.reshape(bs, self.keypoint_num, h, w)
        
        hs = torch.linspace(0, 1, h, device=feat_src.device).type_as(feat_src).unsqueeze(0).unsqueeze(0)
        ws = torch.linspace(0, 1, w, device=feat_tgt.device).type_as(feat_tgt).unsqueeze(0).unsqueeze(0)
        
        kpt_h_src, kpt_w_src = (heatmap_src.sum(3)*hs).sum(2), (heatmap_src.sum(2)*ws).sum(2)
        kpt_h_tgt, kpt_w_tgt = (heatmap_tgt.sum(3)*hs).sum(2), (heatmap_tgt.sum(2)*ws).sum(2)
        
        kpt_loc_src = torch.stack([kpt_h_src, kpt_w_src], dim=2)
        kpt_loc_tgt = torch.stack([kpt_h_tgt, kpt_w_tgt], dim=2)
        
        return kpt_feat_src, kpt_feat_tgt, heatmap_src, heatmap_tgt, kpt_loc_src, kpt_loc_tgt
    
    def recon_image(self, feat_src, feat_tgt, kpt_feat_src, kpt_feat_tgt, kpt_loc_src, kpt_loc_tgt):
        h = self.spatial_size
        w = self.spatial_size
        feat_src = feat_src.reshape(feat_src.size(0), feat_src.size(1), h, w)   
        feat_tgt = feat_tgt.reshape(feat_tgt.size(0), feat_tgt.size(1), h, w)
        
        kpt_gaussin_heatmap_src = gaussian_like_function(kpt_loc_src, h, w, self.gaussian_sigma)
        feat_src_for_recon = torch.einsum("ijkl, ijm->imkl",kpt_gaussin_heatmap_src, kpt_feat_src) + torch.einsum('ijk, il->iljk', (1 - kpt_gaussin_heatmap_src.sum(1)), kpt_feat_src.mean(1))                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                      
        kpt_gaussin_heatmap_tgt = gaussian_like_function(kpt_loc_tgt, h, w, self.gaussian_sigma)
        feat_tgt_for_recon = torch.einsum("ijkl, ijm->imkl",kpt_gaussin_heatmap_tgt, kpt_feat_tgt) + torch.einsum('ijk, il->iljk', (1 - kpt_gaussin_heatmap_tgt.sum(1)), kpt_feat_tgt.mean(1))                                 
                                                                                                              
        img_recon_src = self.reconstructer(feat_src_for_recon)
        img_recon_tgt = self.reconstructer(feat_tgt_for_recon)
                                                                         
        return img_recon_src, img_recon_tgt
        
    def recon_image_with_rotation(self, feat_src, feat_tgt, kpt_feat_src, kpt_feat_tgt, kpt_loc_src, kpt_loc_tgt, gt_r, gt_r_inverse):
        """
        使用旋转信息进行图像重建
        Args:
            feat_src, feat_tgt: 源图像和目标图像特征
            kpt_feat_src, kpt_feat_tgt: 关键点特征
            kpt_loc_src, kpt_loc_tgt: 关键点位置
            gt_r: 源图像到目标图像的旋转矩阵
            gt_r_inverse: 目标图像到源图像的旋转矩阵
        """
        h = self.spatial_size
        w = self.spatial_size
                  
        kpt_gaussin_heatmap_src = gaussian_like_function(kpt_loc_src, h, w, self.gaussian_sigma)
        kpt_gaussin_heatmap_tgt = gaussian_like_function(kpt_loc_tgt, h, w, self.gaussian_sigma)
                   
        r_feat = self.rotation_encoder(gt_r.reshape(gt_r.shape[0], -1))
        r_inverse_feat = self.rotation_encoder(gt_r_inverse.reshape(gt_r_inverse.shape[0], -1))
        
        r_feat = r_feat.view(-1, self.embed_channels, 1, 1)
        r_inverse_feat = r_inverse_feat.view(-1, self.embed_channels, 1, 1)
                  
        feat_src_for_recon = torch.einsum("ijkl,ijm->imkl", kpt_gaussin_heatmap_src, kpt_feat_src) +\
                            torch.einsum('ijk,il->iljk', (1 - kpt_gaussin_heatmap_src.sum(1)), kpt_feat_src.mean(1))
        feat_tgt_for_recon = torch.einsum("ijkl,ijm->imkl", kpt_gaussin_heatmap_tgt, kpt_feat_tgt) +\
                            torch.einsum('ijk,il->iljk', (1 - kpt_gaussin_heatmap_tgt.sum(1)), kpt_feat_tgt.mean(1))
          
        r_feat = r_feat.expand_as(feat_src_for_recon)
        r_inverse_feat = r_inverse_feat.expand_as(feat_tgt_for_recon)
        
        fused_feat_src = self.fusion_block(torch.cat([feat_src_for_recon, r_feat], dim=1))
        fused_feat_tgt = self.fusion_block(torch.cat([feat_tgt_for_recon, r_inverse_feat], dim=1))
                
        img_recon_src_r = self.reconstructer(fused_feat_src)
        img_recon_tgt_r = self.reconstructer(fused_feat_tgt)
        
        return img_recon_src_r, img_recon_tgt_r  
    
    def reg_dense_conf(self, x, mode):
        mode, vmin, vmax = mode
        if mode == 'exp':
            return vmin + x.exp().clip(max=vmax-vmin)
    
    def predict_kpt3d_for_kpts(self, kpt_feat_src, kpt_feat_tgt, kpt_loc_src, kpt_loc_tgt):                                       
        kpt_loc_src_spatial = torch.round(kpt_loc_src * (1024 - 1)).to(torch.long)
        kpt_loc_tgt_spatial = torch.round(kpt_loc_tgt * (1024 - 1)).to(torch.long)
        for idx, blk_src in enumerate(self.keypoint_cross_blocks):
            if idx == 0:                                                      
                kpt_feat_src_, _ = blk_src(kpt_feat_src, kpt_feat_tgt, kpt_loc_src_spatial, kpt_loc_tgt_spatial)
            else:                                    
                kpt_feat_src_, _ = blk_src(kpt_feat_src_, kpt_feat_tgt, kpt_loc_src_spatial, kpt_loc_tgt_spatial)
                
        for idx, blk_src in enumerate(self.keypoint_cross_blocks):
            if idx == 0:                                                 
                kpt_feat_tgt_, _ = blk_src(kpt_feat_tgt, kpt_feat_src, kpt_loc_tgt_spatial, kpt_loc_src_spatial)                                                                                                  
            else:                                    
                kpt_feat_tgt_, _ = blk_src(kpt_feat_tgt_, kpt_feat_src, kpt_loc_tgt_spatial, kpt_loc_src_spatial)
        kpt3d_prediction_target = self.kpt3d_head(kpt_feat_tgt_)
        kpt3d_dir_target, kpt3d_weight_target, depth_tgt = (
            kpt3d_prediction_target[:, :, :3],
            kpt3d_prediction_target[:, :, 3],
            kpt3d_prediction_target[:, :, 4],
        )
        kpt3d_weight_target = kpt3d_weight_target.sigmoid()                                                                                                                      
        kpt3d_depth_target = self.kpt3d_depth_base + torch.tanh(depth_tgt)
        kpt3d_prediction_source = self.kpt3d_head(kpt_feat_src_)
        kpt3d_dir_source, kpt3d_weight_source, depth_src = (
            kpt3d_prediction_source[:, :, :3],
            kpt3d_prediction_source[:, :, 3],
            kpt3d_prediction_source[:, :, 4],
        )
        kpt3d_weight_source = kpt3d_weight_source.sigmoid()                                                                                                  
        kpt3d_depth_source = self.kpt3d_depth_base + torch.tanh(depth_src)
                                                                                     
        return kpt3d_dir_target, kpt3d_dir_source, kpt3d_weight_target, kpt3d_weight_source, kpt3d_depth_target, kpt3d_depth_source

    def forward(self, img_src, img_tgt, gt_r=None, gt_r_inverse=None, current_epoch=None, total_epochs=None):
        h, w = img_src.shape[2:]
        bs = img_src.shape[0]
       
        feat_src, feat_tgt = self.encoding(img_src, img_tgt)
        img_mask_src = self.mask_predictor(feat_src)                  
        img_mask_tgt = self.mask_predictor(feat_tgt)
        img_mask_src_small = img_mask_src.mean(dim=-1).reshape(bs, 1, -1)
        img_mask_tgt_small = img_mask_tgt.mean(dim=-1).reshape(bs, 1, -1)
        feat_src = feat_src * img_mask_src_small
        feat_tgt = feat_tgt * img_mask_tgt_small
        img_mask_src = unpatchify(img_mask_src, patch_size=self.patch_size, channels=1)
        img_mask_tgt = unpatchify(img_mask_tgt, patch_size=self.patch_size, channels=1)

        kpt_feat_src, kpt_feat_tgt, heatmap_src, heatmap_tgt, kpt_loc_src, kpt_loc_tgt = self.get_kpt(feat_src, feat_tgt)                      
        
        img_recon_src, img_recon_tgt = self.recon_image(feat_src, feat_tgt, kpt_feat_src, kpt_feat_tgt, kpt_loc_src, kpt_loc_tgt)

        kpt_loc_src_pixel = (kpt_loc_src * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()                             
        kpt_loc_tgt_pixel = (kpt_loc_tgt * torch.tensor([h - 1, w - 1]).to(kpt_loc_src.device)).long()
        batch_indices = torch.arange(bs).view(-1, 1).expand(-1, self.keypoint_num).to(kpt_loc_src.device)         
        kpt_mask_src = img_mask_src.squeeze(1)[batch_indices, kpt_loc_src_pixel[..., 0], kpt_loc_src_pixel[..., 1]]                 
        kpt_mask_tgt = img_mask_tgt.squeeze(1)[batch_indices, kpt_loc_tgt_pixel[..., 0], kpt_loc_tgt_pixel[..., 1]]
        kpt3d_dir_target, kpt3d_dir_source, kpt3d_weight_target, kpt3d_weight_source, kpt3d_depth_target, kpt3d_depth_source = self.predict_kpt3d_for_kpts(kpt_feat_src, kpt_feat_tgt, kpt_loc_src.detach(), kpt_loc_tgt.detach())                                          
        kpt_loc_tgt_norm = (kpt_loc_tgt - 0.5) * 2                
        kpt_loc_src_norm = (kpt_loc_src - 0.5) * 2
        kpt3d_anchor_target = torch.concat([kpt_loc_tgt_norm.detach().flip(dims=[-1]), kpt3d_depth_target.unsqueeze(-1)], dim=-1)
        kpt3d_anchor_source = torch.concat([kpt_loc_src_norm.detach().flip(dims=[-1]), kpt3d_depth_source.unsqueeze(-1)], dim=-1)

        kpt3d_anchor_target = kpt3d_anchor_target / torch.norm(kpt3d_anchor_target, p=2, dim=2, keepdim=True)
        kpt3d_anchor_source = kpt3d_anchor_source / torch.norm(kpt3d_anchor_source, p=2, dim=2, keepdim=True)
        kpt3d_dir_target = kpt3d_dir_target / torch.norm(kpt3d_dir_target, p=2, dim=2, keepdim=True)
        kpt3d_dir_source = kpt3d_dir_source / torch.norm(kpt3d_dir_source, p=2, dim=2, keepdim=True)

        pred_delta_R_inverse = self.svd_head(kpt3d_anchor_target, kpt3d_dir_target, kpt_mask_src, kpt3d_weight_target)
        pred_delta_R = self.svd_head(kpt3d_anchor_source, kpt3d_dir_source, kpt_mask_tgt, kpt3d_weight_source)

        if self.training and gt_r is not None:
            if current_epoch <= 150:
                img_recon_src_r, img_recon_tgt_r = self.recon_image_with_rotation(
                    feat_src, feat_tgt, kpt_feat_src, kpt_feat_tgt, 
                    kpt_loc_src, kpt_loc_tgt, gt_r, gt_r_inverse
                )
            else:                                   
                img_recon_src_r, img_recon_tgt_r = self.recon_image_with_rotation(
                    feat_src.detach(), feat_tgt.detach(), kpt_feat_src.detach(), kpt_feat_tgt.detach(), 
                    kpt_loc_src.detach(), kpt_loc_tgt.detach(), pred_delta_R, pred_delta_R_inverse
                )
        if not self.training:
            self.feat_src = feat_src
            self.feat_tgt = feat_tgt
            self.kpt_feat_src = kpt_feat_src
            self.kpt_feat_tgt = kpt_feat_tgt
            self.kpt_loc_src = kpt_loc_src
            self.kpt_loc_tgt = kpt_loc_tgt
        if self.training:
            return img_recon_src, img_recon_tgt, img_recon_src_r, img_recon_tgt_r, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source               
        else:
            return img_recon_src, img_recon_tgt, kpt_loc_src, kpt_loc_tgt, img_mask_src, img_mask_tgt, kpt3d_anchor_target, kpt3d_dir_target, kpt3d_anchor_source, kpt3d_dir_source, pred_delta_R, pred_delta_R_inverse, kpt3d_weight_target, kpt3d_weight_source               
      
class SVDHead(nn.Module):
    def __init__(self, transport=None, mask="both"):
        super().__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.transport = transport
        self.temp_score = 0.1
        self.temp_mask = 1.0
        self.bin_score = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.skh_iters = 3
        self.log_optimal_transport = log_optimal_transport
        self.mask = mask

    def forward(self, *input):
        src_pts3d = input[0]            
        tgt_pts3d = input[1]                             
        kpt_src_mask = input[2]
        kpt_tgt_mask = input[3]
        pts3d_in_src = src_pts3d
        pts3d_in_tgt = tgt_pts3d
        batch_size = src_pts3d.size(0)
        mask = kpt_tgt_mask
        mask_diag = torch.stack([torch.diag(mask[i]) for i in range(batch_size)], dim=0)               
        H = torch.matmul(torch.matmul(pts3d_in_src.transpose(2, 1), mask_diag), pts3d_in_tgt)               
        R = []
        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.transpose(2, 1).contiguous())
        r_det = torch.det(r)
        for i in range(pts3d_in_src.size(0)):
            if r_det[i] < 0:
                R.append(torch.matmul(torch.matmul(v[i], self.reflect), u[i].transpose(1, 0).contiguous()))
            else:
                R.append(r[i])
        R = torch.stack(R, dim=0)
                                                                                                           
        return R

def gaussian_like_function(kp_loc, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp_loc[:, :, 0], height)
    wm = squared_diff(kp_loc[:, :, 1], width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + eps).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm

def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm

