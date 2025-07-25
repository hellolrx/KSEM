#代码分为两个部分，上部分完成中毒数据的聚类（预测与标签匹配的逻辑），下部分完成干净数据的聚类（预测与标签不匹配的逻辑）。
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# from torch import nn
# import config
# from utils import tools
# import shutup
# from multiprocessing import freeze_support
#
# shutup.please()
#
# # 定义数据集类
# class CIFAR10Dataset(Dataset):
#     def __init__(self, img_dir, labels_file, transform=None):
#         self.img_dir = img_dir
#         self.labels = torch.load(labels_file, weights_only=False)  # 加载 .pth 文件
#         self.transform = transform
#
#         # 获取所有图片文件名，并按数值排序
#         self.img_files = sorted(
#             [f for f in os.listdir(img_dir) if f.endswith('.png')],
#             key=lambda x: int(x.split('.')[0])  # 按文件名中的数字排序
#         )
#
#         # 检查文件和标签数量是否匹配
#         if len(self.img_files) != len(self.labels):
#             raise ValueError(f"Number of images ({len(self.img_files)}) does not match number of labels ({len(self.labels)})")
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         img_file = self.img_files[idx]
#         img_path = os.path.join(self.img_dir, img_file)
#
#         # 加载图片
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image not found: {img_path}")
#         image = Image.open(img_path).convert('RGB')
#
#         # 获取标签
#         label = self.labels[idx].item()  # 从 Tensor 中取整数标签
#
#         # 数据增强
#         if self.transform:
#             image = self.transform(image)
#
#         # 返回图片、标签和文件名中的索引
#         img_idx = int(img_file.split('.')[0])  # 提取原始索引
#         return image, label, img_idx
#
# # 数据预处理（用于模型推理）
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
# ])
#
# def main():
#     # 加载数据集
#     img_dir = r"D:\KSEM\ResNet\Trojan\8k_5k\poison_set\image"
#     labels_file = r"D:\KSEM\ResNet\Trojan\8k_5k\poison_set\labels\labels.pth"
#     dataset = CIFAR10Dataset(img_dir, labels_file, transform=transform)
#
#     # 创建 DataLoader
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  # num_workers=0 避免多进程问题
#
#     # 加载模型
#     num_classes = 10  # CIFAR-10 有 10 个类别
#     arch = config.arch['cifar10']  # 从 config 中获取架构
#     model = arch(num_classes=num_classes)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #model = nn.DataParallel(model)
#     model = model.to(device)
#
#     # 加载训练好的模型权重
#     model_path = r"D:\KSEM\ResNet\poision_model\Trojan_38k.pth"
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         print(f"Loaded model weights from {model_path}")
#     else:
#         raise FileNotFoundError(f"Model file not found at {model_path}")
#
#     model.eval()  # 设置为评估模式
#
#     # 输出文件路径（改为 .pth 格式）
#     output_file = r"D:\KSEM\ResNet\txt_recoerd\cross_Trojan_38k_poison_vaild5k.pth"
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#
#     # 记录预测正确的索引
#     matched_indices = []
#
#     # 验证数据和标签是否对应（通过模型预测）
#     with torch.no_grad():  # 不计算梯度，节省内存
#         for batch_idx, (images, labels, img_indices) in enumerate(tqdm(data_loader, desc="Validating dataset")):
#             images = images.to(device)
#             labels = labels.to(device)
#             img_indices = img_indices.to(device)  # 原始文件名索引
#
#             # 模型预测
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)  # 获取预测类别
#
#             # 对比预测和真实标签，并输出结果
#             for i in range(len(labels)):
#                 img_idx = img_indices[i].item()  # 原始文件名索引
#                 pred = predicted[i].item()       # 预测类别
#                 true = labels[i].item()          # 真实标签
#                 is_match = pred == true          # 是否匹配
#
#                 # 输出每个样本的预测信息
#                 print(f"Index: {img_idx}, Predicted: {pred}, True: {true}, Match: {is_match}")
#
#                 if is_match:  # 如果预测正确
#                     matched_indices.append(img_idx)
#
#     # 将匹配的索引保存为 .pth 文件
#     matched_indices_tensor = torch.LongTensor(matched_indices)
#     torch.save(matched_indices_tensor, output_file)
#
#     print(f"Found {len(matched_indices)} matched indices out of {len(dataset)} total samples.")
#     print(f"Matched indices saved to {output_file}")
#
# if __name__ == '__main__':
#     freeze_support()  # 添加对 Windows 多进程的支持
#     main()

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch import nn
import config
from utils import tools
import shutup
from multiprocessing import freeze_support

shutup.please()

# 定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = torch.load(labels_file, weights_only=False)  # 加载 .pth 文件
        self.transform = transform

        # 获取所有图片文件名，并按数值排序
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])  # 按文件名中的数字排序
        )

        # 检查文件和标签数量是否匹配
        if len(self.img_files) != len(self.labels):
            raise ValueError(f"Number of images ({len(self.img_files)}) does not match number of labels ({len(self.labels)})")

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
        label = self.labels[idx].item()  # 从 Tensor 中取整数标签

        # 数据增强
        if self.transform:
            image = self.transform(image)

        # 返回图片、标签和文件名中的索引
        img_idx = int(img_file.split('.')[0])  # 提取原始索引
        return image, label, img_idx

# 数据预处理（用于模型推理）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
])

def main():
    # 加载数据集
    img_dir = r"D:\KSEM\ResNet\Trojan\13k_4k\poison_set\image"
    labels_file = r"D:\KSEM\ResNet\Trojan\13k_4k\poison_set\labels\labels.pth"
    dataset = CIFAR10Dataset(img_dir, labels_file, transform=transform)

    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)  # num_workers=0 避免多进程问题

    # 加载模型
    num_classes = 10  # CIFAR-10 有 10 个类别
    arch = config.arch['cifar10']  # 从 config 中获取架构
    model = arch(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = nn.DataParallel(model)
    model = model.to(device)

    # 加载训练好的模型权重
    model_path = r"D:\KSEM\ResNet\poision_model\Trojan_38k.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.eval()  # 设置为评估模式

    # 输出文件路径（改为 .pth 格式）
    output_file = r"D:\KSEM\ResNet\txt_recoerd\cross_Trojan_38k_poison_vaild4k.pth"  # 修改文件名以反映“不匹配”逻辑
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 记录预测不匹配的索引
    mismatched_indices = []

    # 验证数据和标签是否对应（通过模型预测）
    with torch.no_grad():  # 不计算梯度，节省内存
        for batch_idx, (images, labels, img_indices) in enumerate(tqdm(data_loader, desc="Validating dataset")):
            images = images.to(device)
            labels = labels.to(device)
            img_indices = img_indices.to(device)  # 原始文件名索引

            # 模型预测
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取预测类别

            # 对比预测和真实标签，并输出结果
            for i in range(len(labels)):
                img_idx = img_indices[i].item()  # 原始文件名索引
                pred = predicted[i].item()       # 预测类别
                true = labels[i].item()          # 真实标签
                is_match = pred == true          # 是否匹配

                # 输出每个样本的预测信息
                print(f"Index: {img_idx}, Predicted: {pred}, True: {true}, Match: {is_match}")

                if not is_match:  # 如果预测不匹配
                    mismatched_indices.append(img_idx)

    # 将不匹配的索引保存为 .pth 文件
    mismatched_indices_tensor = torch.LongTensor(mismatched_indices)
    torch.save(mismatched_indices_tensor, output_file)

    print(f"Found {len(mismatched_indices)} mismatched indices out of {len(dataset)} total samples.")
    print(f"Mismatched indices saved to {output_file}")

if __name__ == '__main__':
    freeze_support()  # 添加对 Windows 多进程的支持
    main()