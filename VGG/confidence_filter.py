import torch
import numpy as np
from torchvision import transforms
from utils.tools import IMG_Dataset
from tqdm import tqdm
from utils import resnet
import os

# 定义数据变换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 创建毒化测试数据加载器
poisoned_set = IMG_Dataset(
    data_dir=r"D:\Fight-Poison-With-Poison\blend_poision\data",
    label_path=r"D:\Fight-Poison-With-Poison\blend_poision\labels",
    transforms=data_transform
)

# 加载的VGG模型路径
model_path = r"D:\KSEM\VGG\poision_model\badnet_31k.pth"

# 索引文件路径
indices_file = r"D:\KSEM\VGG\badnet\9k_7k\poison_set\labels\indices.pth"

# 自动选择设备，如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 .pth 文件中的索引
indices = torch.load(indices_file, weights_only=False)
if isinstance(indices, torch.Tensor):
    indices = indices.tolist()  # 转换为列表
print(f"Loaded {len(indices)} indices from {indices_file}, First 10: {indices[:10]}")

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
        try:
            # 从数据集中加载对应索引的图片
            sample, _ = poisoned_set[idx]
            sample = sample.unsqueeze(0).to(device)  # 增加 batch 维度并移动到设备
            output = model(sample)
            probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            confidence = np.max(probabilities)

            # 记录置信度和对应的索引
            confidence_list.append((idx, confidence))
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

# 按置信度从高到低排序
confidence_list.sort(key=lambda x: x[1], reverse=True)

# 计算前 80% 的数量
top_80_percent_num = int(len(confidence_list) * 0.8)

# 获取置信度前 80% 的索引
top_indices = [idx for idx, _ in confidence_list[:top_80_percent_num]]

# 输出满足条件的样本数量
print(f"Number of samples in the top 80% confidence: {len(top_indices)}")

# 将满足条件的样本索引保存为 .pth 文件
output_file = r"D:\KSEM\VGG\txt_recoerd\confidence_top80_Badnet.pth"
top_indices_tensor = torch.LongTensor(sorted(top_indices))  # 按索引排序并转换为 tensor
torch.save(top_indices_tensor, output_file)

print(f"Top 80% confidence indices have been saved to {output_file}.")