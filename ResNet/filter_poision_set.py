#数据分割，利用23行的索引可以把20行的目标数据分成26，28行的干净数据和含毒数据，并加入了检查机制，确保全程数据与标签数量一致，索引文件可以追踪触发器的数量
import os
import shutil
from PIL import Image
from utils.tools import IMG_Dataset
from torchvision import transforms
from tqdm import tqdm  # 导入 tqdm 库
import torch
import shutup

shutup.please()

# 定义数据变换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 定义路径
data_dir = r"D:\KSEM\ResNet\Trojan\8k_5k\poison_set\image"
label_path = r"D:\KSEM\ResNet\Trojan\8k_5k\poison_set\labels\labels.pth"

index_file = r"D:\KSEM\ResNet\txt_recoerd\cross_Trojan_38k_poison_vaild5k.pth"

# 输出路径
output_image_dir_clean = r"D:\KSEM\ResNet\Trojan\2k_2k\clean_set\image"
output_label_dir_clean = r"D:\KSEM\ResNet\Trojan\2k_2k\clean_set\labels"
output_image_dir_poison = r"D:\KSEM\ResNet\Trojan\2k_2k\poison_set\image"
output_label_dir_poison = r"D:\KSEM\ResNet\Trojan\2k_2k\poison_set\labels"

# 创建输出目录
os.makedirs(output_image_dir_clean, exist_ok=True)
os.makedirs(output_label_dir_clean, exist_ok=True)
os.makedirs(output_image_dir_poison, exist_ok=True)
os.makedirs(output_label_dir_poison, exist_ok=True)

# 读取需要剔除的图片索引（从 .pth 文件加载）
indices_to_remove = torch.load(index_file, weights_only=False)
if isinstance(indices_to_remove, torch.Tensor):
    indices_to_remove = set(indices_to_remove.tolist())
else:
    indices_to_remove = set(indices_to_remove)
print(f"Loaded {len(indices_to_remove)} indices from {index_file}")

# 获取所有存在的图片文件名和索引
all_image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
all_indices = [int(f.split('.')[0]) for f in all_image_files]  # 从文件名提取索引
print(f"Found {len(all_indices)} images in {data_dir}, First 10: {all_indices[:10]}")

# 加载标签文件
labels = torch.load(label_path, weights_only=False)
if len(labels) != len(all_indices):
    raise ValueError(f"Number of labels ({len(labels)}) does not match number of images ({len(all_indices)})")
print(f"Loaded {len(labels)} labels from {label_path}")

# 初始化新的标签和索引列表
new_labels_clean = []
new_labels_poison = []
indices_clean = []
indices_poison = []

# 遍历所有存在的图片索引，使用 tqdm 显示进度条
for i, idx in tqdm(enumerate(all_indices), desc="Processing dataset", total=len(all_indices)):
    # 图片路径
    image_name = f"{idx}.png"
    image_path = os.path.join(data_dir, image_name)

    # 检查图片是否存在（冗余检查）
    if os.path.exists(image_path):
        # 获取对应的标签（假设标签顺序与文件名排序一致）
        label = labels[i]

        # 如果当前索引在需要剔除的索引中，则属于“含毒数据集”
        if idx in indices_to_remove:
            # 复制图片到含毒数据集目录
            output_image_path = os.path.join(output_image_dir_poison, image_name)
            shutil.copy(image_path, output_image_path)

            # 保存标签和索引到含毒数据集
            new_labels_poison.append(label.item())
            indices_poison.append(idx)  # 索引取图片文件名数字部分
        else:
            # 复制图片到干净数据集目录
            output_image_path = os.path.join(output_image_dir_clean, image_name)
            shutil.copy(image_path, output_image_path)

            # 保存标签和索引到干净数据集
            new_labels_clean.append(label.item())
            indices_clean.append(idx)  # 索引取图片文件名数字部分

# 将干净数据集标签转换为 torch.LongTensor 并保存为 .pth 文件
new_labels_clean_tensor = torch.LongTensor(new_labels_clean)
output_label_path_clean = os.path.join(output_label_dir_clean, "labels.pth")
torch.save(new_labels_clean_tensor, output_label_path_clean)

# 将干净数据集索引转换为 torch.LongTensor 并保存为 .pth 文件
indices_clean_tensor = torch.LongTensor(indices_clean)
output_indices_path_clean = os.path.join(output_label_dir_clean, "indices.pth")
torch.save(indices_clean_tensor, output_indices_path_clean)

# 将含毒数据集标签转换为 torch.LongTensor 并保存为 .pth 文件
new_labels_poison_tensor = torch.LongTensor(new_labels_poison)
output_label_path_poison = os.path.join(output_label_dir_poison, "labels.pth")
torch.save(new_labels_poison_tensor, output_label_path_poison)

# 将含毒数据集索引转换为 torch.LongTensor 并保存为 .pth 文件
indices_poison_tensor = torch.LongTensor(indices_poison)
output_indices_path_poison = os.path.join(output_label_dir_poison, "indices.pth")
torch.save(indices_poison_tensor, output_indices_path_poison)

# 打印保存信息
print(f"Filtered clean dataset saved to {output_image_dir_clean}, labels: {output_label_path_clean}, indices: {output_indices_path_clean}")
print(f"Poison dataset saved to {output_image_dir_poison}, labels: {output_label_path_poison}, indices: {output_indices_path_poison}")

# 验证数量是否匹配
clean_img_count = len(os.listdir(output_image_dir_clean))
poison_img_count = len(os.listdir(output_image_dir_poison))
clean_label_count = len(new_labels_clean)
poison_label_count = len(new_labels_poison)
clean_indices_count = len(indices_clean)
poison_indices_count = len(indices_poison)

print(f"\nClean dataset: {clean_img_count} images, {clean_label_count} labels, {clean_indices_count} indices")
print(f"Poison dataset: {poison_img_count} images, {poison_label_count} labels, {poison_indices_count} indices")
total_processed = clean_img_count + poison_img_count
if (clean_img_count == clean_label_count == clean_indices_count and
    poison_img_count == poison_label_count == poison_indices_count and
    total_processed == len(all_indices)):
    print(f"Data, labels, and indices match in number for both datasets! Total processed: {total_processed}")
else:
    print(f"Warning: Data, labels, or indices do not match in number! Total processed: {total_processed} vs Expected: {len(all_indices)}")