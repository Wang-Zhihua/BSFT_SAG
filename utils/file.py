
import os
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import  csv
from operator import itemgetter
root = '/data3/lusc/patch_20x_s256_rgb/features/resnet50/'
result = []
save_path = './slides.csv'
slides = os.listdir(root)
for j in range(len(slides)):
    result.append([slides[j]])
'''
for i in range(len(fold)):
    fold[i] = int(fold[i])
    if fold[i] in folds:
        slide = os.listdir(root+str(fold[i]))
        for j in range(len(slide)):
            result.append([slide[j]])'''
with open(save_path,"w",encoding="utf-8",newline='') as f:
    writer=csv.writer(f)
    writer.writerows(result)
f.close()