import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .drop import DropPath
from functools import partial

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2, qkv_bias=False, drop_rate=0.1, drop_path=.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Multi_Head_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio, drop=drop_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Bottleneck_Fusion(nn.Module):

    def __init__(self, dim=64, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.3, num_classes=1):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.fusion_token = nn.Parameter(torch.randn(1, 2, dim))
        
        self.image_token = nn.Parameter(torch.randn(1, 1, dim))
        self.image_fusion = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
        #self.image_fusion1 = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
        self.image_norm = norm_layer(dim)
        self.image_head = nn.Linear(dim, num_classes)
        self.image_act = nn.Sigmoid()
        
        self.clinic_token = nn.Parameter(torch.randn(1, 1, dim))
        self.clinic_fusion = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
        self.clinic_fusion1 = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
        self.clinic_norm = norm_layer(dim)
        self.clinic_head = nn.Linear(dim, num_classes)
        self.clinic_act = nn.Sigmoid()

    def forward(self, image_feat, clinic_feat):
        b, n, _ = image_feat.shape
        image_token = self.image_token.expand(b, -1, -1)
        fusion_token = self.fusion_token.expand(b, -1, -1)
        b, n, _ = clinic_feat.shape
        clinic_token = self.clinic_token.expand(b, -1, -1)

        clinic_feat = torch.cat((clinic_token, clinic_feat, fusion_token), dim=1)
        clinic_feat = self.clinic_fusion(clinic_feat)
        #clinic_attn = (clinic_attn[0,0,0,1:18]+clinic_attn[0,1,0,1:18]+clinic_attn[0,2,0,1:18]+clinic_attn[0,3,0,1:18])*0.25
        fusion_token = clinic_feat[:, -2:]
        image_feat = torch.cat((image_token, image_feat, fusion_token), dim=1)
        image_feat = self.image_fusion(image_feat)
        #image_attn = (image_attn[0,0,0,1:11]+image_attn[0,1,0,1:11]+image_attn[0,2,0,1:11]+image_attn[0,3,0,1:11])*0.25
        
        fusion_token = image_feat[:, -2:]
        clinic_feat = clinic_feat[:, :-2]
        clinic_feat = torch.cat((clinic_feat, fusion_token), dim=1)
        clinic_feat = self.clinic_fusion1(clinic_feat)
        '''
        fusion_token = clinic_feat[:, -2:]
        image_feat = image_feat[:, :-2]
        image_feat = torch.cat((image_feat, fusion_token), dim=1)
        image_feat = self.image_fusion1(image_feat)'''
        
        image_cls_token = image_feat[:, 0]
        image_cls_token = self.image_norm(image_cls_token)
        image_pred = self.image_head(image_cls_token)
        image_pred = self.image_act(image_pred)
        clinic_cls_token = clinic_feat[:, 0]
        clinic_cls_token = self.clinic_norm(clinic_cls_token)
        clinic_pred = self.clinic_head(clinic_cls_token)
        clinic_pred = self.clinic_act(clinic_pred)
        
        pred = (image_pred + clinic_pred) / 2
        return pred

class Clinic_Branch(nn.Module):

    def __init__(self, clinic_dim=17, dim=64, mlp_ratio=2, drop_rate=0.3, cluster_num=10):
        super().__init__()
        self.patch_to_embedding = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, clinic_dim + 1, dim))
        self.dropout = nn.Dropout(drop_rate)
        self.block = Block(dim=dim, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)       
    
    def forward(self, clinic):
        x = torch.transpose(clinic, 1, 0)
        x = self.patch_to_embedding(x)
        x = x.unsqueeze(0)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        clinic_feat = self.block(x)
        return clinic_feat

class Image_Branch(nn.Module):

    def __init__(self, patch_dim=512, dim=64, mlp_ratio=2, drop_rate=0.3, cluster_num=10):
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cluster_num=cluster_num
        self.siamese = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        self.dropout = nn.Dropout(drop_rate)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, cluster_num + 1, dim))
        self.block = Block(dim=dim, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
    def forward(self, x):
        feats = []
        for i in range(self.cluster_num):
            rep = x[i]
            rep = self.patch_to_embedding(rep)
            rep = rep.squeeze(0)
            A = self.siamese(rep)
            A = torch.transpose(A, 1, 0)  # KxN
            rep = torch.mm(A, rep)  # KxL
            feats.append(rep)

        h = torch.cat(feats)
        b = h.size(0)
        d = h.size(1)
        x = h.view(b, d)
        x = x.unsqueeze(0)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        image_feat = self.block(x)
        return image_feat

class Muti_Modal(nn.Module):

    def __init__(self, cluster_num=10, num_classes=1, patch_dim=512, clinic_dim=16, dim=64, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.3):
        super().__init__()
        self.clinic_branch = Clinic_Branch(clinic_dim=clinic_dim, dim=dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate, cluster_num=cluster_num)
        self.image_branch = Image_Branch(patch_dim=patch_dim, dim=dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate, cluster_num=cluster_num)
        self.fusion = Bottleneck_Fusion(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, num_classes=num_classes)
    
    def forward(self, x, clinic):
        clinic_feat = self.clinic_branch(clinic)
        image_feat = self.image_branch(x)
        pred = self.fusion(image_feat, clinic_feat)

        return pred