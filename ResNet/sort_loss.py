import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import resnet
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载模型
model_path = r'D:\KSEM\ResNet\poision_model\cifar10_Blend_seed0.pth'
model = resnet.ResNet18(num_classes=10)
model.load_state_dict(torch.load(model_path))
# 将模型迁移到指定设备（GPU 或 CPU）
model = model.to(device)
model.eval()

# 2. 加载图片和标签
img_dir = r"D:\KSEM\ResNet\sort_loss\img"
label_path = r"D:\KSEM\ResNet\sort_loss\labels\labels.txt"

# 假设标签文件每行一个数字，对应每张图片的标签
with open(label_path, 'r') as f:
    labels = [int(line.strip()) for line in f.readlines()]

# 图像预处理（与训练时相同）
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 获取文件夹中所有图片文件的名称（按数字排序）
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))  # 假设文件名为数字（0.png, 1.png, ...）

# 指定索引列表
indices = [
    231, 422, 473, 547, 1034, 2331, 2936, 3363, 3871, 4113, 4205, 4709, 5348, 5431, 5473, 5629, 5833, 6430, 6471, 6723,
    7363, 7887, 8095, 8308, 8595, 9172, 9466, 9740, 10453, 10960, 11131, 11936, 12545, 12581, 13285, 13338, 13616, 14011,
    14105, 14598, 14656, 14772, 14818, 15238, 15245, 15307, 15775, 15867, 15987, 16124, 16221, 17249, 17413, 17569, 17784,
    17818, 18084, 18343, 19355, 20178, 20356, 21661, 22019, 22515, 22572, 22600, 23169, 23357, 23631, 23900, 24469, 24723,
    25060, 25405, 25794, 26190, 26295, 26367, 26582, 26804, 26862, 27750, 28264, 28559, 28734, 29014, 29063, 29433, 30055,
    30536, 30637, 30733, 31245, 31662, 32151, 32793, 33178, 33265, 33782, 34306, 34572, 34731, 34813, 34817, 35097, 36099,
    36101, 36424, 36806, 36983, 37288, 37895, 38472, 38886, 38945, 39097, 39214, 39463, 39663, 40041, 40937, 41272, 41412,
    41660, 41667, 42008, 42113, 42164, 42403, 42860, 43560, 43736, 44135, 44446, 44600, 45160, 45464, 45470, 45486, 45906,
    46712, 47051, 47448, 48367, 48893, 49277, 49516, 49902, 49950, 49961
]

# 3. 计算损失值并筛选
loss_function = nn.CrossEntropyLoss()
satisfy_indices = []
all_losses = []  # 存储所有数据的损失值
specific_losses = []  # 存储指定索引的损失值

# 使用 tqdm 包装循环，添加进度条
with torch.no_grad():
    for img_name, label in tqdm(zip(img_files, labels), total=len(img_files), desc="Processing images"):
        img_path = os.path.join(img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = data_transform(image).unsqueeze(0)
            # 将数据迁移到指定设备（GPU 或 CPU）
            image = image.to(device)
            label = torch.tensor([label]).to(device)

            output = model(image)
            loss = loss_function(output, label)

            # 判断损失值是否满足筛选条件（低于 0.001 或高于 10）
            if loss.item() < 0.01 or loss.item() > 12:
                # 提取图片名称中的数字部分作为索引
                idx = int(img_name.split('.')[0])  # 获取文件名中的数字部分作为索引
                satisfy_indices.append(idx)

            # 检查当前索引是否在指定索引列表中，如果是则输出损失值
            if int(img_name.split('.')[0]) in indices:  # 获取文件名中的数字部分作为索引
                print(f"Index {img_name.split('.')[0]}, Loss: {loss.item():.4f}")
                specific_losses.append(loss.item())
            all_losses.append(loss.item())

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# 将满足条件的索引写入文件
with open('D:/KSEM/ResNet/txt_recoerd/loss.txt', 'w') as file:
    for index in satisfy_indices:
        file.write(str(index) + '\n')

print(f"满足条件的索引已保存到 loss.txt 文件中")

# 计算直方图
bins = np.arange(0, 26, 1)
hist_all, _ = np.histogram(all_losses, bins=bins)
hist_specific, _ = np.histogram(specific_losses, bins=bins)

# 绘制叠放的柱形图
plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], hist_all, width=1, color='blue', label='All Data')
plt.bar(bins[:-1], hist_specific, width=1, color='red', bottom=hist_all, label='Specific Indices')
plt.title('Loss Distribution')
plt.xlabel('Loss Value')
plt.xticks(np.arange(0, 30, 1))
plt.ylabel('Number')
plt.legend()
plt.show()
