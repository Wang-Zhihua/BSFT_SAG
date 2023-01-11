import os
import csv
import torch
import random
import numpy as np
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn import metrics
from torch.autograd import Variable
from torchvision import datasets, transforms
from dataloader import Bags
import torch.nn as nn
import torch.nn.functional as F
from model.drop import DropPath
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
class Fusion_Block(nn.Module):

    def __init__(self, dim=64, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio, drop=drop_rate)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x, clinic):

        x = self.norm1(x)
        clinic = self.norm1(clinic)
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = clinic.shape
        q = self.q(clinic).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Clinic_Branch(nn.Module):

    def __init__(self, clinic_dim=17, dim=64, mlp_ratio=2, drop_rate=0.3, cluster_num=10, num_classes=1):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.patch_to_embedding = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, clinic_dim, dim))
        self.norm = norm_layer(dim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, clinic):
        x = torch.transpose(clinic, 1, 0)
        x = self.patch_to_embedding(x)
        feat = x.unsqueeze(0)
        b, n, _ = feat.shape
        feat += self.pos_embedding[:, :(n)]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, feat), dim=1)
        x = self.dropout(x)
        return x

class Image_Branch(nn.Module):

    def __init__(self, patch_dim=512, dim=64, mlp_ratio=2, drop_rate=0.3, cluster_num=10, num_classes=1):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cluster_num=cluster_num
        self.siamese = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        self.dropout = nn.Dropout(drop_rate)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, cluster_num, dim))
        self.norm = norm_layer(dim)
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
        feat = x.unsqueeze(0)
        b, n, _ = feat.shape
        feat += self.pos_embedding[:, :(n)]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, feat), dim=1)
        x = self.dropout(x)
        return x

class Muti_Modal(nn.Module):

    def __init__(self, cluster_num=10, num_classes=1, patch_dim=512, clinic_dim=16, dim=64, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.3):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.image_branch = Image_Branch(patch_dim=patch_dim, dim=dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate, num_classes=num_classes, cluster_num=cluster_num)
        self.clinic_branch = Clinic_Branch(clinic_dim=clinic_dim, dim=dim, drop_rate=drop_rate, num_classes=num_classes, cluster_num=cluster_num)
        self.image_CA = Fusion_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate)
        self.clinic_CA = Fusion_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate)
        self.image_head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )
        self.clinic_head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, clinic):
        c = self.clinic_branch(clinic)
        x = self.image_branch(x)
        image_feat = self.image_CA(x, c)
        clinic_feat = self.clinic_CA(c, x)
        image_pred = self.image_head(image_feat[:, 0])
        clinic_pred = self.clinic_head(clinic_feat[:, 0])
        pred = (image_pred + clinic_pred) / 2
        return pred
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed=0
seed_torch(seed)
lr = 0.0003
num_epochs = 20
batch_size = 2
num_patches = 25000
weight_decay=5e-4
gpu_list = [1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pred_loss(pred, label):
    pred_torch = torch.cat(pred)
    label_torch = torch.cat(label)
    loss = torch.mean(-1. * (label_torch * torch.log(pred_torch)*0.86 + 1.19*(1. - label_torch) * torch.log(1. - pred_torch)))
    return loss

def train(i_fold):
    model = Muti_Modal(cluster_num=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6, last_epoch=-1, verbose=False)
    train_set = Bags(patch_length=num_patches, usage='train', i_fold=i_fold)
    train_loader = data_utils.DataLoader(train_set,batch_size=1,shuffle=True)
    valid_set = Bags(patch_length=num_patches, usage='valid', i_fold=i_fold)
    valid_loader = data_utils.DataLoader(valid_set,batch_size=1)
    train_num, valid_num = train_set.num, valid_set.num
    model.train()
    score = []
    result = []
    for epoch in range(num_epochs):
        label_list, pred_list = [], []
        train_loss = 0.
        valid_loss = 0.
        train_acc, valid_acc, valid_err = torch.Tensor([0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([0, 0, 0])
        for batch_idx,(data, label, _, clinic, _) in enumerate(train_loader,0):
            for i in range(10):
              data[i] = data[i].to(device)  
            clinic = clinic.to(device)
            pred = model(data, clinic)
            l = int(label.item())
            if int(torch.ge(pred, 0.5))==l:
                train_acc[l] += 1
                train_acc[2] += 1
            label = label.to(device)
            label = label.long()
            pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5)
            label_list.append(label)
            pred_list.append(pred)
            if batch_idx%batch_size==batch_size-1 or batch_idx== len(train_loader)-1:
                optimizer.zero_grad()
                loss = pred_loss(pred_list, label_list)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
                label_list, pred_list = [], []
        train_acc = 100. * train_acc / train_num
        train_loss = train_loss / train_num[2] * batch_size
                
        scheduler.step()
        y_test = []
        y_test_pred = []
        model.eval()
        for data, label, _, clinic, _ in valid_loader:
            for i in range(10):
              data[i] = data[i].to(device) 
            clinic = clinic.to(device)
            pred = model(data, clinic)
            l = int(label.item())
            y_test.append(l)
            y_test_pred.append(round(float(pred.item()),4))
            if int(torch.ge(pred, 0.5))==l:
                valid_acc[l] += 1
                valid_acc[2] += 1
            else:
                valid_err[l] += 1
                valid_err[2] += 1
            label = label.to(device)
            label = label.long()
            pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5)
            loss = -1. * (label * torch.log(pred)*0.86 + 1.19*(1. - label) * torch.log(1. - pred))
            valid_loss += loss.cpu().item()
        precision = valid_acc[1] / (valid_acc[1] + valid_err[0])
        recall = valid_acc[1] / (valid_acc[1] + valid_err[1])
        roc_auc = metrics.roc_auc_score(np.array(y_test), np.array(y_test_pred))
        f1_score = 2 * precision * recall / (precision + recall)
        valid_acc = 100. * valid_acc / valid_num
        result.append([y_test, y_test_pred])
        score.append([seed, i_fold, epoch+1, round(float(valid_acc[2].item()),2), round(float(roc_auc.item()),4)*100, 
            round(float(precision.item()),4)*100, round(float(recall.item()),4)*100, round(float(f1_score.item()),4)*100])
        valid_loss = valid_loss / valid_num[2]
        
        print('[{}] Train: {} Loss: {:.6f}  Acc: {:.2f}%  Acc0: {:.2f}%  Acc1: {:.2f}%'.format(i_fold, epoch+1, train_loss, train_acc[2], train_acc[0], train_acc[1]))
        print('[{}] Valid: {} Loss: {:.6f}  Acc: {:.2f}%  Acc0: {:.2f}%  Acc1: {:.2f}%'.format(i_fold, epoch+1, valid_loss, valid_acc[2], valid_acc[0], valid_acc[1]))
        print('[{}] Valid: {} Acc: {:.2f}%  AUC: {:.2f}%  Precision: {:.2f}%  Recall: {:.2f}%  F1-Score: {:.2f}%'.format(
            i_fold, epoch+1, valid_acc[2], roc_auc*100, precision*100, recall*100, f1_score*100))
        print('-' * 100)
        with open('./train_log.txt', 'a') as file0:
            print('[{}] Train: {} Loss: {:.6f}  Acc: {:.2f}%  Acc0: {:.2f}%  Acc1: {:.2f}%'.format(
            i_fold, epoch+1, train_loss, train_acc[2], train_acc[0], train_acc[1]), file=file0)
        with open('./valid_log.txt', 'a') as file0:
            print('[{}] Valid: {} Loss: {:.6f}  Acc: {:.2f}%  Acc0: {:.2f}%  Acc1: {:.2f}%'.format(
            i_fold, epoch+1, valid_loss, valid_acc[2], valid_acc[0], valid_acc[1]), file=file0)
        with open('./score_log.txt', 'a') as file0:
            print('[{}] Valid: {} Acc: {:.2f}%  AUC: {:.2f}%  Precision: {:.2f}% Recall: {:.2f}% F1-Score: {:.2f}%'.format(
            i_fold, epoch+1, valid_acc[2], roc_auc*100, precision*100, recall*100, f1_score*100), file=file0)
    with open("./T/folder_"+str(i_fold)+".csv","w",encoding="utf-8",newline='') as f:
        writer=csv.writer(f)
        writer.writerows(result)
    f.close()
    with open("./score_log.csv","a",encoding="utf-8",newline='') as f:
        writer=csv.writer(f)
        writer.writerows(score)
    f.close()

if __name__ == "__main__":
    k = 5
    for i in range(k):
        print('-'*25, 'folder', i, '-'*25)
        train(i)
        