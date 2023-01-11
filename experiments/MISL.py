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
import torch.nn as nn
import torch.nn.functional as F
from model.drop import DropPath
from functools import partial
from dataloader import Bags
    
class Muti_Modal(nn.Module):

    def __init__(self, cluster_num=10, num_classes=1, patch_dim=512, clinic_dim=16, dim=64, bilinear=True, drop_rate=0.3):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.cluster_num = cluster_num
        self.bilinear = bilinear
        self.image_embedding = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.clinic_embedding = nn.Sequential(
            nn.Linear(clinic_dim, dim),
            nn.ReLU(),
            nn.Dropout(p=drop_rate)
        )
        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        if self.bilinear == True:
            self.bilinear = nn.Bilinear(dim, dim, dim)
        self.head = nn.Linear(dim, num_classes)
        self.norm = norm_layer(dim)
        self.act = nn.Sigmoid()
        
    def masked_softmax(self, x, mask=None):
        mask = mask.squeeze(0)
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask+1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)
        
    def forward(self, x, mask, clinic):
        
        feats = []
        for i in range(self.cluster_num):
            rep = x[i]
            rep = self.image_embedding(rep)
            rep = rep.view(rep.size()[0], -1)
            feats.append(rep)
        h = torch.cat(feats)
        b = h.size(0)
        d = h.size(1)
        image_feat = h.view(b, d)
        clinic_feat = self.clinic_embedding(clinic)
        
        if self.bilinear:
            image_feat = image_feat.squeeze(0)
            A = self.attention(image_feat)
            A = torch.transpose(A, 1, 0)  # KxN
            A = self.masked_softmax(A, mask)
            image_feat = torch.mm(A, image_feat)
            feat = self.bilinear(image_feat, clinic_feat)
        else:
            feat = torch.cat((clinic_feat, image_feat), dim=0)
            A = self.attention(feat)
            A = torch.transpose(A, 1, 0)  # KxN
            feat = torch.mm(A, feat)
        feat = self.norm(feat)
        pred = self.head(feat)
        pred = self.act(pred)

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

bilinear = True
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
    model = Muti_Modal(cluster_num=10, bilinear=bilinear).to(device)
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
        for batch_idx,(data, label, mask, clinic, name) in enumerate(train_loader,0):
            for i in range(10):
              data[i] = data[i].squeeze(0)
              data[i] = data[i].t()
              data[i] = data[i].unsqueeze(0)
              data[i] = data[i].unsqueeze(2)
              data[i] = data[i].to(device)
            clinic = clinic.to(device)
            mask = mask.to(device)
            pred = model(data, mask, clinic)
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
        for data, label, mask, clinic, name in valid_loader:
            for i in range(10):
              data[i] = data[i].squeeze(0)
              data[i] = data[i].t()
              data[i] = data[i].unsqueeze(0)
              data[i] = data[i].unsqueeze(2)
              data[i] = data[i].to(device) 
            clinic = clinic.to(device)
            mask = mask.to(device)
            pred = model(data, mask, clinic)
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
        