#数据抽取，用于第二次干净数据聚类，确保模型能对十种干净数据进行分类，保证足够的ACC来筛选触发器
import os
import shutil
import torch
import random
from natsort import natsorted
from collections import Counter
from tqdm import tqdm

# 定义路径
poisoned_set_img_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\3k_7k\poison_set\image"
poisoned_set_label_path = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\3k_7k\poison_set\labels\labels.pth"

output_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\extract"
output_image_dir = os.path.join(output_dir, "image")
output_label_dir = os.path.join(output_dir, "labels")
output_label_path = os.path.join(output_label_dir, "labels.pth")
output_indices_path = os.path.join(output_label_dir, "indices.pth")

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 加载并检查标签
print(f"Loading labels from: {poisoned_set_label_path}")
labels = torch.load(poisoned_set_label_path, weights_only=False)
print(f"Total number of labels: {len(labels)}")
print(f"First 10 labels: {labels[:10].tolist()}")

# 检查图片文件
print(f"\nLoading image files from: {poisoned_set_img_dir}")
img_files = natsorted([f for f in os.listdir(poisoned_set_img_dir) if f.endswith('.png')])
print(f"Total number of image files: {len(img_files)}")
print(f"First 10 image filenames: {img_files[:10]}")

# 对比标签和文件数量
if len(labels) != len(img_files):
    raise ValueError(f"Mismatch detected! Labels: {len(labels)}, Image files: {len(img_files)}")
print("\nLabels and image files match in number!")

# 检查文件索引和标签的对应关系
print("\nChecking correspondence between filenames and labels (first 10 samples):")
filename_to_label = {}
for i, filename in enumerate(img_files):
    file_idx = int(filename.split('.')[0])
    label = labels[i].item()
    filename_to_label[filename] = label
    if i < 10:
        print(f"File: {filename}, Index: {file_idx}, Label: {label}")

# 统计每个类别的数量
print("\nCounting number of samples per class (CIFAR-10, classes 0-9):")
label_counts = Counter(labels.tolist())
total_samples = len(labels)
cifar10_classes = range(10)
for class_idx in cifar10_classes:
    count = label_counts.get(class_idx, 0)
    percentage = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")

# 选择指定类别（1, 4, 5, 6, 8, 9）
target_classes = {2,3}
target_files = [f for f, l in filename_to_label.items() if l in target_classes]
print(f"\nFound {len(target_files)} samples in target classes {target_classes}, First 10: {target_files[:10]}")

# 随机抽取
sample_size = int(len(target_files) * 0.2)
selected_files = random.sample(target_files, sample_size)
selected_files = sorted(selected_files, key=lambda x: int(x.split('.')[0]))  # 按文件名升序排序
print(f"Randomly selected {len(selected_files)} samples (25%) from target classes")

# 初始化输出列表
selected_labels = []
selected_indices = []

# 复制选中的图片并记录标签和索引
for filename in tqdm(selected_files, desc="Copying selected samples"):
    src_path = os.path.join(poisoned_set_img_dir, filename)
    dst_path = os.path.join(output_image_dir, filename)
    shutil.copy2(src_path, dst_path)

    # 获取标签和索引
    label = filename_to_label[filename]
    idx = int(filename.split('.')[0])
    selected_labels.append(label)
    selected_indices.append(idx)

# 将标签和索引保存为 .pth 文件
selected_labels_tensor = torch.LongTensor(selected_labels)
selected_indices_tensor = torch.LongTensor(selected_indices)
torch.save(selected_labels_tensor, output_label_path)
torch.save(selected_indices_tensor, output_indices_path)

# 打印保存信息
print(f"\nSelected dataset saved to {output_image_dir}, labels: {output_label_path}, indices: {output_indices_path}")

# 验证数量是否匹配
img_count = len(os.listdir(output_image_dir))
label_count = len(selected_labels)
indices_count = len(selected_indices)
print(f"\nSelected dataset: {img_count} images, {label_count} labels, {indices_count} indices")
if img_count == label_count == indices_count:
    print("Data, labels, and indices match in number!")
else:
    print("Warning: Data, labels, or indices do not match in number!")

# 验证匹配性
print("\nValidating selected dataset:")
mismatch_count = 0
for i, filename in enumerate(selected_files):
    idx = int(filename.split('.')[0])
    if idx != selected_indices[i].item():
        print(f"Mismatch: Filename {filename} index {idx} does not match {selected_indices[i].item()}")
        mismatch_count += 1
    if filename_to_label[filename] != selected_labels[i].item():
        print(f"Mismatch: Filename {filename} label {filename_to_label[filename]} does not match {selected_labels[i].item()}")
        mismatch_count += 1
if mismatch_count == 0:
    print("Validation passed: All data, labels, and indices match correctly!")
else:
    print(f"Validation failed: Found {mismatch_count} mismatches!")