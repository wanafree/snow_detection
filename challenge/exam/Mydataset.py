"""
dataset for u2net
"""
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, input_size=(320, 320)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = list()
        self.gt_mask = list()
        # self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        # self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.load_data()

    def __len__(self):
        return len(self.samples)

    def load_data(self):
        img_dir_full_path = self.img_dir
        mask_dir_full_path = self.mask_dir
        img_files = os.listdir(img_dir_full_path)

        for img_name in tqdm(img_files):
            img_full_path = os.path.join(img_dir_full_path, img_name)
            base, extend = os.path.splitext(img_name)
            mask_full_path = os.path.join(mask_dir_full_path, base + '.png')
            img = Image.open(img_full_path)
            # img = cv2.resize(img, self.input_size)
            # img2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img2rgb = img2rgb.astype(np.float32)
            # img2norm = (img2rgb - self.mean) / self.std  # 图像归一化

            # # 图像格式改为nchw
            # img2nchw = np.transpose(img2norm, [2, 0, 1]).astype(np.float32)

            gt_mask = Image.open(mask_full_path)

            # gt_mask = gt_mask.convert('L')

            # gt_mask = cv2.imread(mask_full_path)
            # gt_mask = cv2.resize(gt_mask, self.input_size)
            # gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            # gt_mask = gt_mask / 255.  # 图像归一化
            # gt_mask = np.expand_dims(gt_mask, axis=0)

            self.samples.append(img)
            self.gt_mask.append(gt_mask)

        return self.samples, self.gt_mask

    def __getitem__(self, index):
        img = self.samples[index]
        mask = self.gt_mask[index]
        img = self.transform(img)
        mask = self.mask_transform(mask)

        return img, mask
