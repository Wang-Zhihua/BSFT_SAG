import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
import os


class Bags(data_utils.Dataset):
    def __init__(self, patch_length=200, clusters=10, usage='train', i_fold=-1):
        assert usage in ('train', 'valid')
        self.patch_length = patch_length
        self.usage = usage
        self.label_dir = '/home/zhw/BSFT_SAMIL/table/'
        self.label, self.bag, self.mask, self.clinic, self.name = [], [], [], [], []
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
            self.label.append(int(slide_label))
            self.name.append(slide_id)
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
            self.clinic.append(self.clinic_table[slide_id])         
        slide_file.close()
        print('load ' + self.usage + ' dataset:', len(self.bag))

    def __getitem__(self, index):
        label = []
        label.append(self.label[index])
        label = torch.Tensor(label)
        mask = torch.Tensor(self.mask[index])
        clinic = torch.Tensor(self.clinic[index])
        return self.bag[index], label, mask, clinic, self.name[index]

    def __len__(self):
        return len(self.label)