import os
import csv
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn import metrics
from model.drop import DropPath
from torch.autograd import Variable
from torchvision import datasets, transforms
from functools import partial

class Bags(data_utils.Dataset):
    def __init__(self, patch_length=200, clusters=10, usage='train', i_fold=-1):
        assert usage in ('train', 'valid')
        self.patch_length = patch_length
        self.usage = usage
        self.label_dir = '/home/zhw/BSFT_SAMIL/table/'
        self.label, self.bag, self.mask, self.clinic = [], [], [], []
        self.num = torch.Tensor([0, 0, 0])
        self.clus_table = {}
        self.clusters = clusters
        self.patch_num = {}
        cluster_file_path = self.label_dir + '/resnet18_cluster10.csv'
        cluster_file = open(os.path.join(cluster_file_path), 'r')
        cluster_reader = csv.reader(cluster_file)
        for slide_id, n, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 in cluster_reader:
            clus_stat = [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5), float(c6), float(c7), float(c8), float(c9)]
            self.clus_table[slide_id] = clus_stat
            self.patch_num[slide_id] = sum(clus_stat)
        cluster_file.close()
        slide_file_name = 'k_folder/' + str(i_fold) + '/' + self.usage + '.csv'
        slide_file = open(os.path.join(self.label_dir, slide_file_name), 'r')
        slide_reader = csv.reader(slide_file)
        for slide_id, folder, slide_label, _, _ in slide_reader:
            if slide_id == 'ID':
                continue
            self.label.append(int(slide_label))
            self.num[2] += 1
            self.num[int(slide_label)] += 1
            graph = []
            mask = np.ones(self.clusters, dtype=np.float32)
            #self.name.append(slide_id)
            patch_num = self.patch_num[slide_id]
            patch_file_path = './resnet18_slides/' + slide_id + '/' + slide_id + '_cluster10.csv'
            patch_file = open(os.path.join(patch_file_path), 'r')
            patch_reader = csv.reader(patch_file)
            prop = self.clus_table[slide_id]
            prop[clusters-1] = self.patch_length
            for i in range(clusters-1):
                prop[i] = int(float(prop[i]) * self.patch_length / patch_num)
                prop[clusters-1] -= prop[i]
            for i in range(self.clusters):
                g = np.zeros((prop[i],512), dtype=np.float32)
                if prop[i] == 0:
                    mask[i] = 0
                    g = np.zeros((1,512), dtype=np.float32)
                graph.append(g)
            for patch_id, clus in patch_reader:
                clus = int(clus)
                if prop[clus]==0:
                    continue
                prop[clus] -= 1
                j = prop[clus]
                patch_id = patch_id.split('.')[0] + '.npz'
                data = np.load(os.path.join('./resnet18_slides', slide_id, patch_id))
                data = data['feature']
                mu = np.mean(data,axis=0)
                std = np.std(data,axis=0)
                graph[clus][j] = (data - mu)/std
            patch_file.close()
            for i in range(self.clusters):
                graph[i] = torch.Tensor(graph[i])
            self.bag.append(graph)
            self.mask.append(mask)       
        slide_file.close()
        print('load ' + self.usage + ' dataset:', len(self.bag))

    def __getitem__(self, index):
        label = []
        label.append(self.label[index])
        label = torch.Tensor(label)
        mask = torch.Tensor(self.mask[index])
        return self.bag[index], label, mask

    def __len__(self):
        return len(self.label)

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
     
class Image_Branch(nn.Module):

    def __init__(self, patch_dim=512, dim=64, mlp_ratio=2, drop_rate=0.3, num_classes=1, num_patches=200, cluster_num=10):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        self.cluster_num=cluster_num
        self.siamese = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_cls_token = nn.Identity()
        self.block = Block(dim=dim, num_heads=4, mlp_ratio=mlp_ratio, qkv_bias=True, drop_rate=drop_rate)
        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes)
        self.act = nn.Sigmoid()
    
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
        x = self.dropout(x)
        x = self.block(x)
        image_feat_p = self.to_cls_token(x[:, 0])
        image_feat_p = self.norm(image_feat_p)
        image_pred = self.head(image_feat_p)
        image_pred = self.act(image_pred)
        return image_pred

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
num_patches = 250
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
        for batch_idx,(data, label, _) in enumerate(train_loader,0):
            for i in range(10):
              data[i] = data[i].to(device)  
            pred = model(data)
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
        for data, label, _ in valid_loader:
            for i in range(10):
              data[i] = data[i].to(device)
            pred = model(data)
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
        