import os
import sys
import torch
from torchvision import transforms
from utils.tools import IMG_Dataset
from natsort import natsorted  # 用于自然排序文件名
from collections import Counter  # 用于统计类别数量

# 指定标签路径
poisoned_set_img_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\35k\image"
poisoned_set_label_path = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\35k\labels\labels.pth"

# 加载并检查标签
print(f"Loading labels from: {poisoned_set_label_path}")
try:
    labels = torch.load(poisoned_set_label_path, weights_only=False)  # 显式设置 weights_only=False
    print(f"Total number of labels: {len(labels)}")
    print(f"Label type: {type(labels)}")
    print(f"First 10 labels: {labels[:10].tolist()}")
    print(f"Label tensor shape: {labels.shape}")
except Exception as e:
    print(f"Error loading labels: {e}")
    sys.exit(1)

# 检查图片文件
print(f"\nLoading image files from: {poisoned_set_img_dir}")
try:
    img_files = natsorted([f for f in os.listdir(poisoned_set_img_dir) if f.endswith('.png')])
    print(f"Total number of image files: {len(img_files)}")
    print(f"First 10 image filenames: {img_files[:10]}")
except Exception as e:
    print(f"Error loading image files: {e}")
    sys.exit(1)

# 对比标签和文件数量
if len(labels) == len(img_files):
    print("\nLabels and image files match in number!")
else:
    print(f"\nMismatch detected! Labels: {len(labels)}, Image files: {len(img_files)}")

# 检查文件索引和标签的对应关系（前10个样本）
print("\nChecking correspondence between filenames and labels (first 10 samples):")
for i in range(min(10, len(img_files))):
    filename = img_files[i]
    label = labels[i].item()
    file_idx = int(filename.split('.')[0])  # 提取文件名中的数字
    print(f"File: {filename}, Index: {file_idx}, Label: {label}")

# 统计每个类别的数量
print("\nCounting number of samples per class (CIFAR-10, classes 0-9):")
label_counts = Counter(labels.tolist())  # 将张量转换为列表并统计
total_samples = len(labels)

# 定义 CIFAR-10 的所有类别 (0-9)
cifar10_classes = range(10)

# 按类别顺序打印统计结果，包括缺失的类别
for class_idx in cifar10_classes:
    count = label_counts.get(class_idx, 0)  # 如果类别不存在，返回 0
    percentage = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")

# 打印类别总数和样本总数
unique_classes = len(label_counts) if total_samples > 0 else 0
print(f"\nTotal number of unique classes present: {unique_classes}")
print(f"Total number of samples: {total_samples}")