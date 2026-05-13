import csv
import os
import random

import cv2
import torch
import numpy as np
from time import time

from PIL import Image
from scipy import stats
from torch import nn

from torchvision.transforms import transforms
from tqdm import tqdm
from model.mode import U2NET, U2NETP
from model.efficient_netv2 import effnetv2_s

import warnings

warnings.filterwarnings("ignore")

# process of Semantic segmentation
try:
    print('===loading model===')
    model_save_path = 'model/save'
    u2net = U2NET(3, 1)
    effect_net = effnetv2_s()

    num_ftrs = effect_net.classifier.in_features
    effect_net.fc = nn.Linear(num_ftrs, 2)
    # net = U2NETP(3, 1)
    u2net_checkpoint_path = 'model/save/checkpoint_400.pth'
    effect_net_path = 'model/save/best_network.pth'
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(u2net_checkpoint_path, map_location='cuda:0'))
        effect_net.load_state_dict(torch.load(effect_net_path, map_location='cuda:0'))
    else:
        u2net.load_state_dict(torch.load(u2net_checkpoint_path, map_location='cpu'))
        effect_net.load_state_dict(torch.load(effect_net_path, map_location='cpu'))
    u2net.eval()
    effect_net.eval()
    print('===model lode sucessed===')

except Exception as e:
    print('===model load error:{}==='.format(e))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# 图像归一化操作
def img2norm(img_array, input_size):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    _std = np.array(std).reshape((1, 1, 3))
    _mean = np.array(mean).reshape((1, 1, 3))

    img_array = cv2.resize(img_array, input_size)
    norm_img = (img_array - _mean) / _std

    return norm_img


def normPred(d):
    max = torch.max(d)
    min = torch.min(d)
    dn = (d - min) / (max - min)

    return dn


def segmentation(img_path, input_size):
    cost_time = 0

    input_img = cv2.imread(img_path)
    # cv2.imwrite('input.png', input_img)
    ori_h, ori_w = input_img.shape[:2]
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img = img2norm(img, input_size)

    # 转换图像为tensor
    x_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    x_tensor = torch.unsqueeze(x_tensor, 0)

    start_t = time()
    d1, d2, d3, d4, d5, d6, d7 = u2net(x_tensor)
    end_t = time()
    cost_time = end_t - start_t

    pred = d1[:, 0, :, :]
    pred = normPred(pred)
    pred = pred.squeeze().cpu().data.numpy()
    pred_res = pred * 255
    pred_res = cv2.resize(pred_res, (ori_w, ori_h))

    # 将预测出来的结果图片进行颜色反转，作为mask蒙板和原图运算，得到抠图后的背景
    _, img_mask = cv2.threshold(pred_res, 200, 255, cv2.THRESH_BINARY_INV)
    # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('img_mask01.jpg', img_mask)
    img_mask = img_mask.astype(np.uint8)
    # print(img_mask.dtype)
    re_img = cv2.bitwise_and(input_img, input_img, mask=img_mask)
    # cv2.imwrite('img_mask.jpg', re_img)

    return cost_time, re_img



# 将抠图后的背景划分为64*64的小格
def classify(img):
    start_t = time()
    pixel_grid = []
    height, weight, _ = img.shape
    grid_size = 64
    for r in range(0, height, grid_size):
        for c in range(0, weight, grid_size):
            grid_img = img[r:r + grid_size, c:c + grid_size]
            height, width, _ = grid_img.shape
            mask_num = 0

            for y in range(height):
                for x in range(width):

                    pixel = grid_img[y, x]

                    if (pixel == [0, 0, 0]).all():
                        mask_num += 1
            proportion = mask_num / 4096
            if proportion <= 0.2:
                # grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                pixel_grid.append(grid_img)

    # 逐图像预测
    # average = []
    # for img in pixel_grid:
    #     tensor_img = transform_test(img)
    #     output = effect_net(tensor_img.unsqueeze(0))
    #     output = torch.softmax(output.data, dim=1)
    #     max_prob = output.data[:, 1]
    #     max_prob = max_prob.numpy()
    #     average.append(max_prob)

    # 将剩下的背景分为小格后，随机选取50个进行运算，用T检测求取偏差值
    grid_len = len(pixel_grid)
    random_elements = []
    random_nums = random.sample(range(grid_len), 50)
    for index in random_nums:
        random_elements.append(pixel_grid[index])

    average = []
    for img in random_elements:
        tensor_img = transform_test(img)
        output = effect_net(tensor_img.unsqueeze(0))
        output = torch.softmax(output.data, dim=1)
        max_prob = output.data[:, 1]
        max_prob = max_prob.numpy()
        maxprob = round(max_prob.item(), 2)
        average.append(maxprob)

    # for i in average:
    #     print(i)

    # the  t-detection on the right side of the average
    _, right_p = stats.ttest_1samp(average, 0.6, alternative='greater')

    # the  t-detection on the left side of the average
    _, left_p = stats.ttest_1samp(average, 0.5, alternative='less')
    i = 0
    while right_p > 0.01 and left_p > 0.01:
        # random get one img
        remain_elements = np.setdiff1d(range(grid_len), random_nums)
        if len(remain_elements) == 0:
            # raise ValueError("The `remain_elements` list is empty!")
            break
        else:
            one_num = random.choice(remain_elements)
        random_nums.append(one_num)
        add_img = pixel_grid[one_num]
        random_elements.append(add_img)
        #  get the out put of the img
        tensor_img = transform_test(add_img)
        output = effect_net(tensor_img.unsqueeze(0))
        output = torch.softmax(output.data, dim=1)
        max_prob = output.data[:, 1]
        max_prob = max_prob.numpy()
        maxprob = round(max_prob.item(), 2)
        average.append(maxprob)

        # the  t-detection on the right side of the average
        _, right_p = stats.ttest_1samp(average, 0.6, alternative='greater')

        # the  t-detection on the left side of the average
        _, left_p = stats.ttest_1samp(average, 0.5, alternative='less')
        i += 1

    print("经历了{}次循环，左偏差值为{},右偏差值为{}".format(i, right_p, left_p))

    # print(average)
    print(len(average))
    mean_value = np.mean(average)

    end_t = time()
    cost_time = end_t - start_t

    mean_value = f"{mean_value:.2f}"

    return cost_time, mean_value


if __name__ == '__main__':
    csv_save_path = 'result/result.csv'
    img_path = '../../data/u2_snow_data/test/imgs'

    # 创建文件夹及文件并写入表头
    parent_folder = os.path.dirname(csv_save_path)
    os.makedirs(parent_folder, exist_ok=True)
    data_set = ['number', 'img_path', 'time_cost', 'pre_value']

    with open(csv_save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_set)

        # 循环遍历所有图像
        for i, file in enumerate(os.listdir(img_path)):
            full_img_path = os.path.join(img_path, file)
            cost_time_se, mask_img = segmentation(full_img_path, input_size=(160, 160))
            cost_time_classify, value = classify(mask_img)
            cost_time_all = cost_time_se + cost_time_classify
            cost_time_all = f"{cost_time_all:.2f}"
            print(
                "number = {}, img_path = {}, all the time cost = {}, pre_value = {}".format(i, file, cost_time_all,
                                                                                            value))
            data = [i, file, cost_time_all, value]
            writer.writerow(data)
            # 刷新文件缓冲区，从而实时写入磁盘
            csvfile.flush()

        # 执行完毕后，关闭文件
        csvfile.close()






