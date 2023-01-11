import os
import csv
import torch
import random
import numpy as np
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn import metrics
from torch.autograd import Variable
from model.BSFT_AP import Muti_Modal, Discriminator
from torchvision import datasets, transforms

class Bags(data_utils.Dataset):
    def __init__(self, patch_length=25000, clusters=10, usage='train', i_fold=-1):
        assert usage in ('train', 'valid')
        self.patch_length = patch_length
        self.usage = usage
        self.label_dir = '/home/zhw/BSFT_SAMIL/table/'
        self.label, self.bag, self.clinic = [], [], []
        self.name = []
        self.num = torch.Tensor([0, 0, 0])
        self.clus_table = {}
        self.clusters = clusters
        self.clinic_table = {}
        self.patch_num = {}
        clinic_file_path = self.label_dir + '/clinic_feature.csv'
        clinic_file = open(os.path.join(clinic_file_path), 'r')
        clinic_reader = csv.reader(clinic_file)
        for clinic_data in clinic_reader:
            if clinic_data[0]=='id':
                continue
            slide_id = clinic_data[0]
            clinic_feat = []
            for i in range(len(clinic_data)):
                if i == 0:
                    continue
                clinic_feat.append(float(clinic_data[i]))
            self.clinic_table[slide_id] = clinic_feat
        clinic_file.close()
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
            self.name.append(slide_id)
            self.label.append(int(slide_label))
            self.num[2] += 1
            self.num[int(slide_label)] += 1
            patch_num = int(self.patch_num[slide_id])
            prop = self.clus_table[slide_id]
            prop[clusters-1] = self.patch_length
            for i in range(clusters-1):
                prop[i] = int(float(prop[i]) * self.patch_length / patch_num)
                prop[clusters-1] -= prop[i]
            graph = np.zeros((self.patch_length,512), dtype=np.float32)
            if patch_num<self.patch_length:
                graph = np.zeros((patch_num,512), dtype=np.float32)
            patch_file_path = './resnet18_slides/' + slide_id + '/' + slide_id + '_cluster10.csv'
            patch_file = open(os.path.join(patch_file_path), 'r')
            patch_reader = csv.reader(patch_file)
            j = 0
            for patch_id, clus in patch_reader:
                clus = int(clus)
                if prop[clus]==0:
                    continue
                prop[clus] -= 1
                patch_id = patch_id.split('.')[0] + '.npz'
                data = np.load(os.path.join('./resnet18_slides', slide_id, patch_id))
                data = data['feature']
                mu = np.mean(data,axis=0)
                std = np.std(data,axis=0)
                graph[j] = (data - mu)/std
                j += 1 
            patch_file.close()
            graph= torch.Tensor(graph)
            self.bag.append(graph)
            self.clinic.append(self.clinic_table[slide_id])         
        slide_file.close()
        print('load ' + self.usage + ' dataset:', len(self.bag))

    def __getitem__(self, index):
        label = []
        label.append(self.label[index])
        label = torch.Tensor(label)
        clinic = torch.Tensor(self.clinic[index])
        return self.bag[index], label, clinic

    def __len__(self):
        return len(self.label)
        
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed=1
seed_torch(seed)
lr = 0.0003
num_epochs = 30
batch_size = 2
num_patches = 2500
weight_decay=5e-4
gpu_list = [0,1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0)
person = torch.Tensor([-1.]).to(device)
i_label = [0]
c_label = [1]
def pred_loss(pred, label):
    loss = -1. * (label * torch.log(pred)*0.86 + 1.19*(1. - label) * torch.log(1. - pred))
    return loss
    
def discriminator_loss(pred, label):
    label = torch.Tensor(label)
    label = label.to(device)
    label = label.long()
    loss = -1. * (label * torch.log(pred) + (1. - label) * torch.log(1. - pred))
    return loss
    
def train(i_fold):
    train_set = Bags(patch_length=num_patches, usage='train', i_fold=i_fold)
    train_loader = data_utils.DataLoader(train_set,batch_size=1,shuffle=True)
    valid_set = Bags(patch_length=num_patches, usage='valid', i_fold=i_fold)
    valid_loader = data_utils.DataLoader(valid_set,batch_size=1)
    
    model = Muti_Modal(cluster_num=10).to(device)
    discriminator = Discriminator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    optimizer_d = optim.Adam(discriminator.parameters(), lr = lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6, last_epoch=-1, verbose=False)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=num_epochs, eta_min=5e-6, last_epoch=-1, verbose=False)
    train_num, valid_num = train_set.num, valid_set.num
    score = []
    result = []
    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        train_loss, train_loss_p, train_loss_fip, train_loss_fcp, train_loss_cp, train_loss_ip, train_loss_ff, train_loss_cf, train_loss_if, train_loss_d = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        valid_loss, valid_loss_p, valid_loss_fip, valid_loss_fcp, valid_loss_cp, valid_loss_ip, valid_loss_cf, valid_loss_if = 0., 0., 0., 0., 0., 0., 0., 0.
        train_acc, valid_acc, valid_err = torch.Tensor([0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([0, 0, 0])
        for batch_idx,(data, label, clinic) in enumerate(train_loader,0):
            data = data.to(device) 
            clinic = clinic.to(device)
            pred, ip_pred, ig_pred, ip_feat, ig_feat, cp_pred, cg_pred, cp_feat, cg_feat = model(data, clinic)
            l = int(label.item())
            if int(torch.ge(pred, 0.5))==l:
                train_acc[l] += 1
                train_acc[2] += 1
            label = label.to(device)
            label = label.long()
            pred = torch.clamp(pred, min=1e-5, max=1. - 1e-5)
            cp_pred = torch.clamp(cp_pred, min=1e-5, max=1. - 1e-5)
            ip_pred = torch.clamp(ip_pred, min=1e-5, max=1. - 1e-5)
            cg_pred = torch.clamp(cg_pred, min=1e-5, max=1. - 1e-5)
            ig_pred = torch.clamp(ig_pred, min=1e-5, max=1. - 1e-5)
            loss_p = pred_loss(pred, label)
            loss_fcp = pred_loss(cg_pred, label)
            loss_fip = pred_loss(ig_pred, label)
            loss_cp = pred_loss(cp_pred, label)
            loss_ip = pred_loss(ip_pred, label)
            loss_cf = cosine_loss(cp_feat, cg_feat.detach(), person)
            loss_if = cosine_loss(ip_feat, ig_feat.detach(), person)
            #loss = loss_p + loss_fp + loss_cp + loss_ip + loss_cf + loss_if
            loss = loss_p + loss_fcp + loss_fip + loss_cp + loss_ip + loss_cf + loss_if
            for param in discriminator.parameters():
                param.requires_grad = False
            i_out = discriminator(ig_feat)
            c_out = discriminator(cg_feat)
            loss_ff = 0.5*(discriminator_loss(i_out, c_label) + discriminator_loss(c_out, i_label))
            train_loss += loss.cpu().item()
            train_loss_p += loss_p.cpu().item()
            train_loss_fip += loss_fip.cpu().item()
            train_loss_fcp += loss_fcp.cpu().item()
            train_loss_cp += loss_cp.cpu().item()
            train_loss_ip += loss_ip.cpu().item()
            train_loss_cf += loss_cf.cpu().item()
            train_loss_if += loss_if.cpu().item()
            train_loss_ff += loss_ff.cpu().item()
            loss = loss / batch_size
            loss_ff = loss_ff / batch_size
            loss.backward(retain_graph=True)
            loss_ff.backward()
            for param in discriminator.parameters():
                param.requires_grad = True
            i_out = discriminator(ig_feat.detach())
            c_out = discriminator(cg_feat.detach())
            loss_d = 0.5*(discriminator_loss(i_out, i_label) + discriminator_loss(c_out, c_label))
            train_loss_d += loss_d.cpu().item()
            loss_d = loss_d / batch_size
            loss_d.backward()
            if batch_idx%batch_size==batch_size-1 or batch_idx== len(train_loader)-1:
                optimizer.step()
                optimizer_d.step()
                optimizer.zero_grad()
                optimizer_d.zero_grad()  
        train_acc = 100. * train_acc / train_num
        train_loss = train_loss / train_num[2]
        train_loss_p = train_loss_p / train_num[2]
        train_loss_fip = train_loss_fip / train_num[2]
        train_loss_fcp = train_loss_fcp / train_num[2]
        train_loss_cp = train_loss_cp / train_num[2]
        train_loss_ip = train_loss_ip / train_num[2]
        train_loss_cf = train_loss_cf / train_num[2]
        train_loss_if = train_loss_if / train_num[2]
        train_loss_ff = train_loss_ff / train_num[2]
        train_loss_d = train_loss_d / train_num[2]
        scheduler.step()
        scheduler_d.step()
        y_test = []
        y_test_pred = []
        model.eval()
        for data, label, clinic in valid_loader:
            data = data.to(device) 
            clinic = clinic.to(device)
            pred, ip_pred, ig_pred, ip_feat, ig_feat, cp_pred, cg_pred, cp_feat, cg_feat = model(data, clinic)
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
            cp_pred = torch.clamp(cp_pred, min=1e-5, max=1. - 1e-5)
            ip_pred = torch.clamp(ip_pred, min=1e-5, max=1. - 1e-5)
            cg_pred = torch.clamp(cg_pred, min=1e-5, max=1. - 1e-5)
            ig_pred = torch.clamp(ig_pred, min=1e-5, max=1. - 1e-5)
            loss_cf = cosine_loss(cp_feat, cg_feat.detach(), person)
            loss_if = cosine_loss(ip_feat, ig_feat.detach(), person)
            loss_p = pred_loss(pred, label)
            loss_fcp = pred_loss(cg_pred, label)
            loss_fip = pred_loss(ig_pred, label)
            loss_cp = pred_loss(cp_pred, label)
            loss_ip = pred_loss(ip_pred, label)
            #loss = loss_p + loss_fp + loss_cp + loss_ip + loss_cf + loss_if
            loss = loss_p + loss_fcp + loss_fip + loss_cp + loss_ip + loss_cf + loss_if
            valid_loss += loss.cpu().item()
            valid_loss_p += loss_p.cpu().item()
            valid_loss_fip += loss_fip.cpu().item()
            valid_loss_fcp += loss_fcp.cpu().item()
            valid_loss_cp += loss_cp.cpu().item()
            valid_loss_ip += loss_ip.cpu().item()
            valid_loss_cf += loss_cf.cpu().item()
            valid_loss_if += loss_if.cpu().item()
        precision = valid_acc[1] / (valid_acc[1] + valid_err[0])
        recall = valid_acc[1] / (valid_acc[1] + valid_err[1])
        roc_auc = metrics.roc_auc_score(np.array(y_test), np.array(y_test_pred))
        f1_score = 2 * precision * recall / (precision + recall)
        valid_acc = 100. * valid_acc / valid_num
        result.append([y_test, y_test_pred])
        score.append([seed, i_fold, epoch+1, round(float(valid_acc[2].item()),2), round(float(roc_auc.item()),4)*100, 
            round(float(precision.item()),4)*100, round(float(recall.item()),4)*100, round(float(f1_score.item()),4)*100])
        valid_loss = valid_loss / valid_num[2]
        valid_loss_p = valid_loss_p / valid_num[2]
        valid_loss_fip = valid_loss_fip / valid_num[2]
        valid_loss_fcp = valid_loss_fcp / valid_num[2]
        valid_loss_cp = valid_loss_cp / valid_num[2]
        valid_loss_ip = valid_loss_ip / valid_num[2]
        valid_loss_cf = valid_loss_cf / valid_num[2]
        valid_loss_if = valid_loss_if / valid_num[2]
        print('[{}] Train: {} Loss_ff: {:.6f} Loss_d: {:.6f}'.format(
            i_fold, epoch+1, train_loss_ff, train_loss_d))
        print('[{}] Train: {} Loss_fcp: {:.6f} Loss_fip: {:.6f} Loss_cp: {:.6f} Loss_ip: {:.6f} Loss_cf: {:.6f} Loss_if: {:.6f}'.format(
            i_fold, epoch+1, train_loss_fcp, train_loss_fip, train_loss_cp, train_loss_ip, train_loss_cf, train_loss_if))
        print('[{}] Train: {} Loss: {:.6f}    Loss_p: {:.6f}  Acc: {:.2f}%    Acc0: {:.2f}%   Acc1: {:.2f}%'.format(
            i_fold, epoch+1, train_loss, train_loss_p, train_acc[2], train_acc[0], train_acc[1]))
        print('[{}] Valid: {} Loss_fcp: {:.6f} Loss_fip: {:.6f} Loss_cp: {:.6f} Loss_ip: {:.6f} Loss_cf: {:.6f} Loss_if: {:.6f}'.format(
            i_fold, epoch+1, valid_loss_fcp, valid_loss_fip, valid_loss_cp, valid_loss_ip, valid_loss_cf, valid_loss_if))
        print('[{}] Valid: {} Loss: {:.6f}    Loss_p: {:.6f}  Acc: {:.2f}%    Acc0: {:.2f}%   Acc1: {:.2f}%'.format(
            i_fold, epoch+1, valid_loss,valid_loss_p, valid_acc[2], valid_acc[0], valid_acc[1]))
        print('[{}] Valid: {} Acc: {:.2f}% ROC_AUC: {:.2f}% Precision: {:.2f}% Recall: {:.2f}% F1-Score: {:.2f}%'.format(
            i_fold, epoch+1, valid_acc[2], roc_auc*100, precision*100, recall*100, f1_score*100))
        print('-' * 100)
        with open('./train_log.txt', 'a') as file0:
            print('[{}] Train: {} Loss_ff: {:.6f} Loss_d: {:.6f}'.format(
            i_fold, epoch+1, train_loss_ff, train_loss_d), file=file0)
            print('[{}] Train: {} Loss_fcp: {:.6f} Loss_fip: {:.6f} Loss_cp: {:.6f} Loss_ip: {:.6f} Loss_cf: {:.6f} Loss_if: {:.6f}'.format(
            i_fold, epoch+1, train_loss_fcp, train_loss_fip, train_loss_cp, train_loss_ip, train_loss_cf, train_loss_if), file=file0)
            print('[{}] Train: {} Loss: {:.6f}    Loss_p: {:.6f}  Acc: {:.2f}%    Acc0: {:.2f}%   Acc1: {:.2f}%'.format(
            i_fold, epoch+1, train_loss, train_loss_p, train_acc[2], train_acc[0], train_acc[1]), file=file0)
        with open('./valid_log.txt', 'a') as file0:
            print('[{}] Valid: {} Loss_fcp: {:.6f} Loss_fip: {:.6f} Loss_cp: {:.6f} Loss_ip: {:.6f} Loss_cf: {:.6f} Loss_if: {:.6f}'.format(
            i_fold, epoch+1, valid_loss_fcp, valid_loss_fip, valid_loss_cp, valid_loss_ip, valid_loss_cf, valid_loss_if), file=file0)
            print('[{}] Valid: {} Loss: {:.6f}    Loss_p: {:.6f}  Acc: {:.2f}%    Acc0: {:.2f}%   Acc1: {:.2f}%'.format(
            i_fold, epoch+1, valid_loss,valid_loss_p, valid_acc[2], valid_acc[0], valid_acc[1]), file=file0)
        with open('./score_log.txt', 'a') as file0:
            print('[{}] Valid: {} Acc: {:.2f}% ROC_AUC: {:.2f}% Precision: {:.2f}% Recall: {:.2f}% F1-Score: {:.2f}%'.format(
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
        