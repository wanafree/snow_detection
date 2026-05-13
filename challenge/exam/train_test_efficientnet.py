import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Mydataset_efficient import Mydataset_ef
from torchvision.transforms import transforms
from model.efficient_netv2 import effnetv2_s
from torch.utils.tensorboard import SummaryWriter
from early_stop import EarlyStopping

data_path = '../../data/classify_data_ef'
save_path = 'model/save'

modellr = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter("efficient_logs")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.RandomResizedCrop(150),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = Mydataset_ef(root=data_path, transform=transform, train=True)
dataset_test = Mydataset_ef(root=data_path, transform=transform_test, train=False)

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# train session

criterion = nn.CrossEntropyLoss()
model = effnetv2_s()
num_ftrs = model.classifier.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=modellr)
early_stopping = EarlyStopping(save_path)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    for index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        writer.add_scalar('Loss/train', loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss = sum_loss + print_loss
        if (index + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (index + 1) * len(data), len(train_loader.dataset),
                       100. * (index + 1) / len(train_loader), loss.item()))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            # data = torch.tensor(data)
            output = model(data)
            loss = criterion(output, label)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == label)
            print_loss = loss.data.item()
            test_loss += print_loss
        # correct = correct.data.item()
        acc = correct / total_num
        writer.add_scalar('accuracy/test', acc, epoch)
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
    return test_loss


if __name__ == '__main__':
    for epoch in range(0, EPOCHS):
        adjust_learning_rate(optimizer, epoch)
        train(model, DEVICE, train_loader, optimizer, epoch)
        eval_loss = test(model, DEVICE, test_loader,epoch)
        early_stopping(eval_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    # mode_name = 'efficient_netv2.pth'
    # torch.save(model.state_dict(), os.path.join(save_path, mode_name))
