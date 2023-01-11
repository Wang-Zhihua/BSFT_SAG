import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
import os


clusters = 10
label_dir = '/home/zhw/Siam_SSA/table/'
slide_file_name = 'slide_dir.csv'
slide_file = open(os.path.join(label_dir, slide_file_name), 'r')
slide_reader = csv.reader(slide_file)
result = []
for slide_id, folder, slide_label in slide_reader:
    patch_file_path = '/home/zhw/Siam_SSA/mlp_slides/' + slide_id + '/' + slide_id + '_cluster10.csv'
    patch_file = open(os.path.join(patch_file_path), 'r')
    patch_reader = csv.reader(patch_file)
    clus_stat = np.zeros(clusters)
    
    for _, clus in patch_reader:
        clus = int(clus)
        clus_stat[clus] += 1

    patch_file.close()
    result.append([slide_id,slide_label,clus_stat[0],clus_stat[1],clus_stat[2],clus_stat[3],clus_stat[4],
        clus_stat[5],clus_stat[6],clus_stat[7],clus_stat[8],clus_stat[9]])
    
slide_file.close()
save_path = label_dir + 'mlp_cluster10.csv'
with open(save_path,"w",encoding="utf-8",newline='') as f:
    writer=csv.writer(f)
    writer.writerows(result)
f.close()