import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from tqdm import tqdm  # 导入进度条库
from torchvision import transforms
from utils import resnet
from utils.tools import IMG_Dataset  # 导入IMG_Dataset
import shutup

shutup.please()

# 数据预处理
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 创建毒化测试数据加载器
poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\Trojan_poision\data",
                           label_path=r"D:\Fight-Poison-With-Poison\Trojan_poision\labels",
                           transforms=data_transform)

# 直接使用整个数据集进行训练
trainloader = DataLoader(poisoned_set, batch_size=128, shuffle=True, num_workers=2)

# 定义模型
model = resnet.ResNet18(num_classes=10)

# 使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    train_pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch + 1} Training')
    for i, (inputs, labels) in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_pbar.set_postfix(loss=running_loss / (i + 1))

# 测试函数
def test(loader):
    model.eval()
    correct = 0
    total = 0
    test_pbar = tqdm(enumerate(loader), total=len(loader), desc='Testing')
    with torch.no_grad():
        for i, (inputs, labels) in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_pbar.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    return accuracy

# 主函数
def main():
    num_epochs = 60  # 固定训练60轮次
    best_accuracy = 0.0
    save_path = r'D:\KSEM\ResNet\poision_model'  # 指定保存路径

    for epoch in range(num_epochs):
        train(epoch)
        accuracy = test(trainloader)  # 使用 trainloader 来计算测试准确度

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(save_path, 'cifar10_Trojan_seed0.pth'))

        scheduler.step()

    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    freeze_support()  # 在Windows上需要调用
    main()
