import os
import shutil
from utils.tools import IMG_Dataset
from PIL import Image
from torchvision import transforms

# 定义路径
data_dir = r"D:\Fight-Poison-With-Poison\Badnet_poision\data"
label_path = r"D:\Fight-Poison-With-Poison\blend_poision\labels"
indices_path = r'D:\KSEM\VGG\txt_recoerd\all_correct_badnet_intersection.txt'
img_dir = 'D:/KSEM/VGG/sort_loss/img'  # 目标图片存储路径
new_label_path = 'D:/KSEM/VGG/sort_loss/labels'  # 新的标签路径

# 确保目标文件夹存在
os.makedirs(img_dir, exist_ok=True)
os.makedirs(new_label_path, exist_ok=True)

# 读取索引
with open(indices_path, 'r') as f:
    indices = [int(line.strip()) for line in f.readlines()]

# 使用 IMG_Dataset 加载数据
poisoned_set = IMG_Dataset(data_dir=data_dir, label_path=label_path, transforms=None)

# 初始化新的标签列表
labels = []

# 遍历索引，直接复制图片而不改变其颜色
for idx in indices:
    image, label = poisoned_set[idx]  # 获取图片和标签
    img_name = f"{idx}.png"  # 假设保存图片的名字为索引名
    img_path = os.path.join(data_dir, img_name)

    # 直接复制图片，不进行任何转换
    if os.path.exists(img_path):
        output_img_path = os.path.join(img_dir, img_name)
        shutil.copy(img_path, output_img_path)

    # 保存标签
    labels.append(label)

# 5. 保存标签到文件
with open(os.path.join(new_label_path, 'labels.txt'), 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"数据已成功提取并保存到 {img_dir} 和 {new_label_path}")
