import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils.vgg import vgg16_bn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载模型
model_path = r'D:\KSEM\VGG\poision_model\badnet_31k.pth'
model = vgg16_bn(num_classes=10)
model.load_state_dict(torch.load(model_path, weights_only=True))
# 将模型迁移到指定设备（GPU 或 CPU）
model = model.to(device)
model.eval()

# 2. 加载图片和标签
img_dir = r"D:\KSEM\VGG\sort_loss\img"
label_path = r"D:\KSEM\VGG\sort_loss\labels\labels.pth"

# 加载标签（从 .pth 文件加载）
try:
    labels = torch.load(label_path, weights_only=False).tolist()  # 加载为列表，假设是标量标签（如 [0, 2, 3, 7, 9]）
    print(f"Loaded {len(labels)} labels from {label_path}")
    print(f"First 10 labels: {labels[:10]}")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit(1)

# 图像预处理（与训练时相同）
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 获取文件夹中所有图片文件的名称（按数字排序）
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))  # 假设文件名为数字（0.png, 1.png, ...）

# 验证图片数量和标签数量是否匹配
if len(img_files) != len(labels):
    raise ValueError(f"Number of images ({len(img_files)}) does not match number of labels ({len(labels)})")

# 指定索引列表
indices = [
    46, 101, 221, 863, 1458, 2655, 3327, 3373, 3407, 3428, 3489, 3738, 4507, 4772, 4790, 5155, 5478, 5977, 6104, 6580,
    6908, 7075, 7928, 8267, 8561, 9331, 9387, 9731, 9990, 10605, 11819, 12401, 12483, 12619, 12723, 13264, 13503, 14101,
    15695, 15770, 16295, 16441, 17732, 19516, 20341, 20365, 20583, 20632, 20967, 21431, 21679, 22996, 23124, 23250, 23358,
    23440, 23460, 24062, 24530, 24614, 24955, 25209, 25815, 25926, 26037, 26080, 26461, 26469, 26550, 26586, 26620, 27003,
    27153, 27326, 27617, 27982, 28102, 28885, 28903, 29011, 29812, 29906, 30032, 30156, 30512, 30698, 31016, 31473, 31643,
    31714, 32175, 32262, 32374, 32434, 32691, 32983, 33167, 33812, 34245, 35018, 35272, 35808, 36560, 36938, 37368, 37812,
    37821, 37902, 38075, 38226, 38477, 39257, 39686, 39834, 40038, 40177, 40281, 40318, 40439, 40573, 40774, 41117, 41375,
    41584, 41692, 42028, 42425, 42768, 42919, 43569, 43685, 43791, 43898, 44005, 44393, 46026, 46087, 46544, 46960, 47083,
    47198, 47768, 48122, 48128, 48304, 48331, 48412, 49447, 49708, 49783
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

            # 判断损失值是否满足筛选条件（低于 0.001 或高于 12）
            if loss.item() < 0.001 or loss.item() > 12:
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

# 将满足条件的索引保存为 .pth 文件
output_indices_path = r'D:\KSEM\VGG\txt_recoerd\badnet_loss.pth'
torch.save(torch.tensor(satisfy_indices, dtype=torch.long), output_indices_path)
print(f"满足条件的索引已保存到 {output_indices_path} 文件中")

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