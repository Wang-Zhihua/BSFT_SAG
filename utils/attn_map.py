import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
import os
import cv2
import numpy as np

attn_table = {}
slide_list = []
attn_file = open('/home/zhw/BSFT_SAMIL/attn2/ip_attn.csv', 'r')
attn_reader = csv.reader(attn_file)
for slide_id, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 in attn_reader:
    attn_stat = [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5), float(c6), float(c7), float(c8), float(c9)]
    attn_table[slide_id] = attn_stat
    slide_list.append(slide_id)
attn_file.close()
slide_file = open('/home/zhw/BSFT_SAMIL/table/slide_dir.csv', 'r')
slide_reader = csv.reader(slide_file)
for slide_id, folder, _, _, _, num_h, num_w in slide_reader:
    if slide_id not in slide_list:
        continue
    patch_file_path = './resnet18_slides/' + slide_id + '/' + slide_id + '_cluster10.csv'
    patch_file = open(os.path.join(patch_file_path), 'r')
    patch_reader = csv.reader(patch_file)
    attn = attn_table[slide_id]
    attn_map = np.zeros((int(num_h),int(num_w)), dtype=np.float32)
    im = cv2.imread('/data1/thumdnail/' + folder + '/' + slide_id + '.png')
    #im = resize(im)
    for patch_id, clus in patch_reader:
        clus = int(clus)
        h = int( patch_id.split('_')[1])
        w =  patch_id.split('_')[2]
        w = int(w.split('.')[0])
        attn_map[h][w] = attn[clus]

    amap = cv2.cvtColor(attn_map, cv2.COLOR_RGB2BGR)
    new_map = cv2.resize(amap, (im.shape[1], im.shape[0]))
    normed_mask = new_map / np.max(new_map)
    normed_mask = np.uint8(255 * (1-normed_mask))
    normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    #normed_mask = im*0.6 + normed_mask * 1.0
    normed_mask = cv2.addWeighted(im, 0.6, normed_mask, 1.0, 0)
    cv2.imwrite('./ip_attn_view/' + slide_id + '.png', cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))



        
        
     