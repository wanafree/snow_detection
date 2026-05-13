import datetime
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Mydataset import Mydataset
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.mode import U2NET
from model.mode import U2NETP

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose({
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
})

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')  # BCELoss = - (y * log(p) + (1 - y) * log(1 - p))


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    # 将模型输出和标签数据的数据类型转换为Float
    d0 = d0.float()
    d1 = d1.float()
    d2 = d2.float()
    d3 = d3.float()
    d4 = d4.float()
    d5 = d5.float()
    d6 = d6.float()
    labels_v = labels_v.float()

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #     loss5.data.item(),
    #     loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------
model_name = 'u2net'


def load_data(img_folder, mask_folder, batch_size, input_size):
    train_dataset = Mydataset(img_dir=os.path.join(img_folder, 'train/imgs'),
                              transform=transform,
                              mask_transform=mask_transform,
                              mask_dir=os.path.join(mask_folder, 'train/mask'),
                              input_size=input_size)
    test_dataset = Mydataset(img_dir=os.path.join(img_folder, 'test/imgs'),
                             transform=test_transform,
                             mask_transform=mask_transform,
                             mask_dir=os.path.join(mask_folder, 'test/mask'),
                             input_size=input_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     modellrnew = modellr * (0.1 ** (epoch // 50))
#     print("lr:", modellrnew)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = modellrnew


# ------- 3. training --------
def train_model(epoch_nums, cuda_device, model_save_dir):
    """
    :param epoch_nums: 训练总的epoch
    :param cuda_device: 指定gpu训练
    :param model_save_dir: 模型保存folder
    :return:
    """
    writer = SummaryWriter("logs")
    current_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(current_time, '%Y-%m-%d-%H-%M')
    model_save_dir = os.path.join(model_save_dir, current_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    else:
        pass
    device = torch.device(cuda_device)
    # path = '../../data/u2_snow_data/train/imgs'
    train_loader, test_loader = load_data(img_folder='../../data/u2_snow_data',
                                          mask_folder='../../data/u2_snow_data',
                                          batch_size=32,
                                          input_size=(150, 150))
    # input 3-channels, output 1-channels
    net = U2NET(3, 1)
    # net = U2NETP(3, 1) 轻量化要求

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for epoch in range(0, epoch_nums):
        print("-----start train-----epoch={}".format(epoch))
        run_loss = list()
        run_tar_loss = list()
        net.train()
        # adjust_learning_rate(optimizer, epoch)

        for i, (inputs, gt_masks) in enumerate(tqdm(train_loader, position=0)):
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor)
            inputs, gt_masks = inputs.to(device), gt_masks.to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, gt_masks)

            loss.backward()
            optimizer.step()

            run_loss.append(loss.item())
            run_tar_loss.append(loss2.item())
            writer.add_scalar('Loss/train', loss, epoch)
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        if epoch % 100 == 0:
            checkpoint_name = 'checkpoint_' + str(epoch) + ' ' + str(np.mean(run_loss)) + 'pth'
            torch.save(net.state_dict(), os.path.join(model_save_dir, checkpoint_name))
            print("--model saved:{}--".format(checkpoint_name))
            print("--Train Epoch:{}--".format(epoch))
            print("--Train run_loss:{:.4f}--".format(np.mean(run_loss)))
            print("--Train run_tar_loss:{:.4f}--\n".format(np.mean(run_tar_loss)))

    torch.save(net.state_dict(), os.path.join(model_save_dir, '500.pth'))


if __name__ == '__main__':
    train_model(epoch_nums=500, cuda_device='cuda:0', model_save_dir='model/save')
