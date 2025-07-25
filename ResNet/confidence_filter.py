import torch
import numpy as np
from torchvision import transforms
from utils.tools import IMG_Dataset
from tqdm import tqdm
import re
from utils import resnet
import os

# 定义数据变换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 创建毒化测试数据加载器
# poisoned_set = IMG_Dataset(
#     data_dir=r"D:\Fight-Poison-With-Poison\Badnet_poision\data",
#     label_path=r"D:\Fight-Poison-With-Poison\Badnet_poision\labels",
#     transforms=data_transform
# )

poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\Cifar10_TaCT_22\train\data",
                           label_path=r"D:\Fight-Poison-With-Poison\Cifar10_TaCT_22\train\labels",
                           transforms=data_transform)

# poisoned_set = IMG_Dataset(data_dir="D:\\Fight-Poison-With-Poison\\blend_poision\\data",
#                                label_path=r"D:\Fight-Poison-With-Poison\blend_poision\labels",
#                                transforms=data_transform)

# 加载的VGG模型路径
model_path = r"D:\KSEM\ResNet\poision_model\cifar10_TaCT_seed0.pth"

# 自动选择设备，如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取TXT文件中的索引
indices = []
with open("D:\\KSEM\\VGG\\txt_recoerd\\all_correct_intersection.txt", 'r') as file:
    for line in file:
        numbers = re.findall(r'\d+', line)
        if numbers:
            indices.extend([int(num) for num in numbers])

# 加载模型
model = resnet.ResNet18(num_classes=10)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 初始化置信度列表
confidence_list = []

# 开始遍历指定索引的样本，计算置信度
with torch.no_grad():
    for idx in tqdm(indices, desc=f"Processing samples with model {os.path.basename(model_path)}"):
        # 从数据集中加载对应索引的图片
        sample, _ = poisoned_set[idx]
        sample = sample.unsqueeze(0).to(device)  # 增加 batch 维度并移动到设备
        output = model(sample)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        confidence = np.max(probabilities)

        # 记录置信度和对应的索引
        confidence_list.append((idx, confidence))

# 按置信度从高到低排序
confidence_list.sort(key=lambda x: x[1], reverse=True)

# 计算前 80% 的数量
top_80_percent_num = int(len(confidence_list) * 0.80)

# 获取置信度前 80% 的索引
top_indices = [idx for idx, _ in confidence_list[:top_80_percent_num]]

# 输出满足条件的样本数量
print(f"Number of samples in the top 80% confidence: {len(top_indices)}")

# 将满足条件的样本索引保存到文件
output_file = r"D:\KSEM\ResNet\txt_recoerd\confidence_top80_TaCT_ResNet.txt"
with open(output_file, 'w') as file:
    for idx in sorted(top_indices):  # 按索引排序后写入文件
        file.write(f"{idx}\n")

print(f"Top 80% confidence indices have been saved to {output_file}.")
