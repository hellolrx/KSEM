import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from multiprocessing import freeze_support
import time
from utils.vgg import vgg16_bn
import shutup

shutup.please()

# 定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = torch.load(labels_file, weights_only=False).tolist()  # 加载 .pth 文件并转换为列表
        self.transform = transform

        # 获取所有图片文件名，并按数值排序
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])  # 按文件名中的数字排序
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # 加载图片
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')

        # 获取标签
        label = self.labels[idx]

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载自定义数据集（全部用于训练）
img_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\train_on_clean_set\image"  # 修改为图片文件夹
labels_file = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\train_on_clean_set\labels\labels.pth"  # 修改为 .pth 文件
train_dataset = CIFAR10Dataset(img_dir, labels_file, transform=transform)

# 创建训练数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# 检查 CIFAR-10 数据集是否已经存在
cifar10_data_path = './data/cifar-10-batches-py'
if not os.path.exists(cifar10_data_path):
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
else:
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

# 创建验证数据加载器
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# 加载 VGG-16 模型
model = vgg16_bn(num_classes=10)  # 替换为 VGG-16

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "Loss": running_loss / (total / labels.size(0)),
            "Acc": 100. * correct / total
        })

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# 验证函数
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                "Loss": running_loss / (total / labels.size(0)),
                "Acc": 100. * correct / total
            })

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# 主函数
if __name__ == '__main__':
    freeze_support()

    num_epochs = 60  # 固定训练 60 轮
    model_save_path = r"D:\KSEM\VGG\clean_model\blend1_43k"  # 保存模型路径

    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate()
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # 保存最佳模型
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # 计算训练总时间
    total_time = time.time() - start_time
    print(f"Training complete! Total time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")