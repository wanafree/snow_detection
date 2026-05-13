import os
import cv2 as cv
import torch
import numpy as np
from time import time
from tqdm import tqdm
from model.mode import U2NET, U2NETP

"""
初始化模型加载
"""
try:
    print('===loading model===')
    current_project_path = 'model/save'
    net = U2NET(3, 1)
    # net = U2NETP(3, 1)
    checkpoint_path = os.path.join(current_project_path,
                                   'checkpoint_400.pth')
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    else:
        net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net.eval()
    print('===model lode sucessed===')

except Exception as e:
    print('===model load error:{}==='.format(e))


def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


# 图像归一化操作
def img2norm(img_array, input_size):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    _std = np.array(std).reshape((1, 1, 3))
    _mean = np.array(mean).reshape((1, 1, 3))

    img_array = cv.resize(img_array, input_size)
    norm_img = (img_array - _mean) / _std

    return norm_img


# 归一化预测结果
def normPred(d):
    max = torch.max(d)
    min = torch.min(d)
    dn = (d - min) / (max - min)

    return dn


# test
def inference1folder(img_folder, mask_folder, input_size):
    total_time = list()
    total_dices = list()
    for img_file in tqdm(os.listdir(img_folder), position=0):
        img_full_path = os.path.join(img_folder, img_file)
        base, extend = os.path.splitext(img_file)
        mask_full_path = os.path.join(mask_folder, base + '.png')
        img = cv.imread(img_full_path)
        gt_mask = cv.imread(mask_full_path)
        gt_mask = cv.resize(gt_mask, input_size)
        gt_mask = cv.cvtColor(gt_mask, cv.COLOR_BGR2GRAY)
        gt_mask = gt_mask / 255.

        ori_h, ori_w = img.shape[:2]

        img2rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        norm_img = img2norm(img2rgb, input_size)

        x_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        x_tensor = torch.unsqueeze(x_tensor, 0)

        start_t = time()
        d1, d2, d3, d4, d5, d6, d7 = net(x_tensor)
        end_t = time()

        total_time.append(end_t - start_t)
        pred = d1[:, 0, :, :]
        pred = normPred(pred)
        pred = pred.squeeze().cpu().data.numpy()

        dice_value = dice_coef(pred, gt_mask)
        total_dices.append(dice_value)

        pred_res = pred * 255
        pred_res = cv.resize(pred_res, (ori_w, ori_h))

        cv.imwrite(os.path.join(current_project_path, 'infer_output/', img_file), pred_res)

    print('==inference 1 pic avg cost:{:.4f}ms=='.format(np.mean(total_time) * 1000))
    print('==inference avg dice:{:.4f}=='.format(np.mean(total_dices)))


if __name__ == '__main__':
    test_img = '../../data/u2_snow_data/test/imgs'
    test_mask = '../../data/u2_snow_data/test/mask'
    inference1folder(img_folder=test_img, mask_folder=test_mask, input_size=(150, 150))
