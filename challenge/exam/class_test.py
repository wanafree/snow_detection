from model.efficient_netv2 import effnetv2_s
import os
import random

import cv2
import torch
import numpy as np
from time import time
from scipy import stats
from torch import nn
from PIL import Image

from torchvision.transforms import transforms
from tqdm import tqdm
from model.mode import U2NET, U2NETP
from model.efficient_netv2 import effnetv2_s

effect_net_path = 'model/save/best_network.pth'
effect_net = effnetv2_s()
num_ftrs = effect_net.classifier.in_features
effect_net.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    effect_net.load_state_dict(torch.load(effect_net_path, map_location='cuda:0'))
else:
    effect_net.load_state_dict(torch.load(effect_net_path, map_location='cpu'))

effect_net.eval()
print('===model lode sucessed===')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def classify(img):
    start_t = time()
    pixel_grid = []
    height, weight, _ = img.shape
    grid_size = 64
    mask_num = 0
    for r in range(0, height, grid_size):
        for c in range(0, weight, grid_size):
            grid_img = img[r:r + grid_size, c:c + grid_size]
            height, width, _ = grid_img.shape
            pixel_grid.append(grid_img)
            mask_num += 1

    average = []
    for img in pixel_grid:
        tensor_img = transform_test(img)
        output = effect_net(tensor_img.unsqueeze(0))
        output = torch.softmax(output.data, dim=1)
        _, pred = torch.max(output.data, 1)

        maxprob = output.data[:, 1]
        maxprob = maxprob.numpy()
        maxprob = round(maxprob.item(), 2)
        average.append(maxprob)
        print(maxprob)

    print("len={}".format(mask_num))
    print(np.mean(average))


if __name__ == '__main__':
    path = '../../data/test_data/00239.jpg'

    image = cv2.imread(path)
    classify(image)
