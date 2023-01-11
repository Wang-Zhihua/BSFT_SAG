import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
import os
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from operator import itemgetter
label_dir = '/home/zhw/Siam_SSA/mlp_slides/'
slides = os.listdir(label_dir)
k = 0
for slide_id in slides:
    k += 1
    print('[' + str(k) + ']' + 'load slide: ' + slide_id)
    '''if slide_id not in ['01814656-16','01814656-9']:
        continue'''
    patch_file_path = label_dir + slide_id + '/'
    patches = os.listdir(patch_file_path)
    patch_num = 0
    for patch_id in patches:
        if patch_id.split('.')[1] == 'csv':
            continue
        patch_num += 1
    #patch_num = len(patches)-1
    patch = np.zeros((patch_num,768))
    name = []
    j = 0
    for patch_id in patches:
        if patch_id.split('.')[1] == 'csv':
            continue
        name.append(patch_id)
        data = np.load(os.path.join(patch_file_path, patch_id))
        data = data['feature']
        mu = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        patch[j] = (data - mu)/std
        j += 1
    patch_num = j
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(patch)
    #print(calinski_harabasz_score(patch, y_pred))
    result = []
    for i in range(patch_num):
        result.append([name[i], y_pred[i]])
    result = sorted(result, key=itemgetter(1),reverse=False)
    save_path = patch_file_path + slide_id + '_cluster10.csv'
    with open(save_path,"w",encoding="utf-8",newline='') as f:
        writer=csv.writer(f)
        writer.writerows(result)
    f.close()
