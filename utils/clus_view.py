import os
import csv
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
import cv2
import numpy as np
#vit768v2,resnet18,resnet50v2
size = 8
color_list = [(94,38,18), (255,0,0), (255,128,0), (225,225,0), (0,225,0), (46,139,87), (0,225,225), (0,0,225), (25,25,112),(106,90,205)]
clus_table = {}
cluster_file = open(os.path.join('./table/resnet18v2_cluster10.csv'), 'r')
cluster_reader = csv.reader(cluster_file)
for slide_id, _, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 in cluster_reader:
    clus_stat = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    clus_table[slide_id] = clus_stat
cluster_file.close()
slide_file = open(os.path.join('./table/slide_dir.csv'), 'r')
csv_reader = csv.reader(slide_file)
result = []
#for slide_name, folder, label, patch_num, _ in csv_reader:
for slide_name, folder, label in csv_reader:
    '''if slide_name!='01815665-18':
        continue
    prop = clus_table[slide_name]
    prop[9] = 200
    for i in range(9):
        prop[i] = int(float(prop[i]) * 200 / int(patch_num))
        prop[9] -= prop[i]    '''   
    if folder !='12':
        continue        
    im = cv2.imread('/data1/thumdnail/' + folder + '/' + slide_name + '.png')
    label_file = open(os.path.join('./resnet18v2_slides/' + slide_name + '/' + slide_name + '_cluster10.csv'), 'r')
    csv_reader = csv.reader(label_file)
    for name, clus in csv_reader:
        clus = int(clus)
        '''if prop[clus]==0:
            continue
        prop[clus] -= 1'''
        h = int(name.split('_')[1])
        w = name.split('_')[2]
        w = int(w.split('.')[0])
        h1 = h * size
        h2 = h * size + size - 1
        w1 = w * size
        w2 = w * size + size - 1
        '''if clus>0.8:'''
        cv2.rectangle(im, (w1, h1), (w2, h2), color_list[clus], thickness=-1)
        result.append([name,clus])
    #print(slide_name)
    cv2.imwrite('./cluster_view/' + slide_name + '.png', im)
    label_file.close()
'''with open("./selected_patches.csv","w",encoding="utf-8",newline='') as f:
    writer=csv.writer(f)
    writer.writerows(result)
f.close()
slide_file.close()'''
'''
for slide_name in id:
    im = cv2.imread('/data2/thumdnail/12/' + slide_name + '.png')
    img = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    label_file = open(os.path.join('test.csv'), 'r')
    csv_reader = csv.reader(label_file)
    num = 0
    h0 = im.shape[0]
    w0 = im.shape[1]
    
    area_0 = h0/30*w0/30
    for name, label in csv_reader:
        slide = name.split('_')[0]
        if slide == slide_name:
            if label == '1':
                num += 1
                h = int(name.split('_')[1])
                w = name.split('_')[2]
                w = int(w.split('.')[0])
                h1 = h * size
                h2 = h * size + size - 1
                w1 = w * size
                w2 = w * size + size - 1
                cv2.rectangle(img, (w1, h1), (w2, h2), (255, 255, 255), thickness=-1)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area > area_0):
            c_max.append(cnt)
    for i in range(len(c_max)):
        cv2.drawContours(im, c_max, i, (0, 255, 0), thickness=-1)
    cv2.imwrite('test_check/'+slide_name+'.png', im)
    print(slide_name, num, len(c_max))
    label_file.close()'''