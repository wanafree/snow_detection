"""
dataset for efficient_net
"""
import os
from PIL import Image
import numpy as np
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms


class Mydataset_ef(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        imgs = []
        if train:
            img_full_path = os.path.join(root, 'train')
            for img in os.listdir(root + '/train'):
                imgs.append(img)
        else:
            img_full_path = os.path.join(root, 'test')
            for img in os.listdir(root + '/test'):
                imgs.append(img)
        self.imgs = imgs
        self.img_full_path = img_full_path

    def __getitem__(self, index):
        file_name = self.imgs[index]
        label = file_name.split('.')[-2]
        label = int(label)
        label = torch.tensor(label)
        data = Image.open(os.path.join(self.img_full_path, file_name))
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# if __name__ == '__main__':
#     transform_test = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     data_path = '../../data/classify_data_ef'
#     dataset_train = Mydataset_ef(root=data_path, transform=transform_test, train=True)
#     data, label = dataset_train.__getitem__(1024)
#     print(data.shape)
#
#     print(label)
