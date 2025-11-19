import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=4, dim_ffn=768, dropout=0.0, dropout_attn=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_attn, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(dim_ffn, d_model),
        )

    def forward(self, kpt_query, input_feature):
        query = self.norm1(kpt_query)
        attn_out, attn = self.multihead_attn(query=query, key=input_feature, value=input_feature)
        kpt_query = kpt_query + self.dropout1(attn_out)
        query = self.norm2(kpt_query)
        attn_out, _ = self.self_attn(query, query, query)
        kpt_query = kpt_query + self.dropout2(attn_out)
        query = self.norm3(kpt_query)
        query = self.ffn(query)
        kpt_query = kpt_query + self.dropout3(query)
        return kpt_query, attn

class AttnLayer(nn.Module):
    def __init__(self, cfg=None, block_num=4, d_model=768, num_head=4, dim_ffn=768):
        super().__init__()
        self.block_num = block_num
        self.d_model = d_model
        self.num_head = num_head
        self.dim_ffn = dim_ffn
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None) for _ in range(self.block_num)]
        )

    def forward(self, batch_kpt_query, input_feature):
        input_feature = input_feature.transpose(1, 2)
        attn = None
        for block in self.attn_blocks:
            batch_kpt_query, attn = block(batch_kpt_query, input_feature)
        return batch_kpt_query, attn

class ResNet_Decoder(nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()

        def conv(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(channels_out),
                nn.ReLU(True),
            )

        self.upconv1 = nn.Sequential(conv(in_channels, 256), conv(256, 256))
        self.upconv2 = nn.Sequential(conv(256, 128), conv(128, 128))
        self.upconv3 = nn.Sequential(conv(128, 64), conv(64, 64))
        self.upconv4 = nn.Sequential(conv(64, 32), conv(32, 32))
        self.proj = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1, stride=1), nn.Sigmoid())

    def forward(self, feat):
        feat = F.interpolate(feat, scale_factor=(2, 2), mode='nearest')
        feat = self.upconv1(feat)
        feat = F.interpolate(feat, scale_factor=(2, 2), mode='nearest')
        feat = self.upconv2(feat)
        feat = F.interpolate(feat, scale_factor=(2, 2), mode='nearest')
        feat = self.upconv3(feat)
        feat = F.interpolate(feat, scale_factor=(2, 2), mode='nearest')
        feat = self.upconv4(feat)
        return self.proj(feat)
