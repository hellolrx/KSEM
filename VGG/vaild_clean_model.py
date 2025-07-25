import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import freeze_support
import math
import os
from PIL import Image
from utils.vgg import vgg16_bn
from utils import resnet

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# 下载并加载 CIFAR-10 数据集
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',  # 数据集下载路径
    train=False,    # 加载测试集
    download=True,  # 如果数据集不存在，则自动下载
    transform=transform
)

# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)  # 将 num_workers 设置为 0

# 初始化 VGG16 模型
model = vgg16_bn(num_classes=10)  # 初始化 ResNet-18 模型

# 加载保存的模型权重
model_save_path = r"D:\KSEM\VGG\clean_model\blend1_43k"
model.load_state_dict(torch.load(model_save_path, weights_only=True))  # 使用 weights_only=True 避免安全警告

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 验证函数
def validate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())  # 确保标签是 torch.long 类型

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                "Loss": test_loss / (total / labels.size(0)),
                "Acc": 100. * correct / total
            })

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 主函数
if __name__ == '__main__':
    freeze_support()  # 添加 freeze_support

    # 验证模型
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# 定义数据集类
class PoisonedDataset(Dataset):
    def __init__(self, data_dir, indices, transform=None):
        self.data_dir = data_dir
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据索引获取图片路径
        img_name = f"{self.indices[idx]}.png"
        img_path = os.path.join(self.data_dir, img_name)

        # 加载图片
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image

# 指定索引
# # badnet-60000
# indices = [
#     40, 127, 129, 161, 307, 337, 490, 586, 600, 648, 706, 795, 892, 972, 1181,
#     1212, 1293, 1447, 1506, 1526, 1648, 1730, 1801, 1845, 1862, 1882, 1918,
#     2075, 2179, 2206, 2216, 2218, 2246, 2265, 2310, 2315, 2351, 2407, 2523,
#     2571, 2619, 2641, 2645, 2740, 2777, 2803, 2987, 3091, 3095, 3105, 3134,
#     3173, 3210, 3283, 3546, 3559, 3574, 3592, 3603, 3729, 3746, 3881, 3917,
#     4078, 4154, 4224, 4378, 4380, 4396, 4537, 4593, 4762, 4903, 4984, 5078,
#     5169, 5281, 5397, 5410, 5455, 5582, 5635, 5663, 5825, 5868, 5914, 5916,
#     6097, 6309, 6335, 6365, 6367, 6392, 6651, 6686, 6902, 6975, 6986, 6993,
#     6994, 7141, 7198, 7298, 7321, 7591, 7610, 7633, 7659, 7795, 7852, 7874,
#     7897, 7955, 8083, 8136, 8139, 8196, 8223, 8225, 8227, 8229, 8266, 8269,
#     8443, 8445, 8495, 8525, 8534, 8636, 8661, 8679, 8781, 8849, 9169, 9244,
#     9300, 9381, 9454, 9468, 9571, 9592, 9599, 9686, 9702, 9737, 9825, 9856,
#     9895, 9913, 9968
# ]

#blend
indices = [
    35, 129, 166, 214, 377, 476, 541, 550, 621, 645, 740, 825, 943, 1044, 1096, 1099,
    1117, 1456, 1459, 1478, 1505, 1552, 1575, 1584, 1604, 1758, 1780, 1790, 1794,
    1799, 1826, 1828, 1901, 1944, 1947, 2009, 2063, 2116, 2203, 2234, 2277, 2350,
    2482, 2492, 2513, 2605, 2650, 2840, 2884, 2886, 2985, 3023, 3141, 3166, 3232,
    3236, 3249, 3263, 3284, 3301, 3415, 3459, 3520, 3527, 3552, 3674, 3819, 3857,
    3893, 3931, 4085, 4121, 4199, 4200, 4253, 4523, 4536, 4553, 4810, 4827, 4844,
    4850, 4917, 4954, 4969, 5016, 5048, 5052, 5130, 5263, 5307, 5376, 5436, 5447,
    5589, 5636, 5648, 5680, 5709, 6042, 6075, 6112, 6119, 6122, 6130, 6248, 6343,
    6427, 6570, 6679, 6722, 6750, 6861, 6988, 7009, 7037, 7045, 7047, 7063, 7094,
    7144, 7181, 7326, 7331, 7576, 7806, 8059, 8151, 8182, 8214, 8242, 8310, 8431,
    8441, 8450, 8504, 8668, 8826, 8858, 8883, 8929, 8978, 9018, 9292, 9378, 9494,
    9714, 9804, 9937, 9975
]

#TaCT
# indices = [
#     66, 104, 114, 134, 414, 490, 572, 753, 759, 844, 887, 997, 1005, 1013, 1048,
#     1098, 1131, 1158, 1174, 1212, 1229, 1335, 1412, 1480, 1564, 1569, 1583, 1711,
#     1769, 1811, 1923, 1930, 2063, 2080, 2179, 2206, 2238, 2274, 2620, 2694, 2712,
#     2740, 2764, 2813, 2838, 2911, 2986, 3027, 3144, 3153, 3175, 3449, 3485, 3488,
#     3520, 3528, 3629, 3645, 3702, 3715, 3772, 3895, 3963, 4128, 4131, 4144, 4170,
#     4189, 4212, 4250, 4320, 4379, 4424, 4475, 4607, 4647, 4736, 4848, 4850, 4895,
#     4897, 4912, 4978, 5090, 5135, 5347, 5353, 5394, 5409, 5971, 6048, 6166, 6184,
#     6279, 6281, 6300, 6372, 6431, 6492, 6536, 6597, 6704, 6728, 6742, 6767, 6817,
#     6820, 6940, 7008, 7173, 7243, 7274, 7276, 7316, 7376, 7473, 7480, 7555, 7583,
#     7623, 7626, 7731, 7862, 7878, 7894, 7946, 8037, 8215, 8222, 8238, 8248, 8347,
#     8391, 8549, 8663, 8674, 8751, 9185, 9232, 9535, 9601, 9609, 9626, 9694, 9696,
#     9794, 9808, 9906, 9935, 9974
# ]
#Trojan
# indices = [
# 6, 64, 95, 292, 323, 351, 467, 616, 634, 666,
# 741, 809, 815, 874, 1053, 1079, 1166, 1181, 1198, 1200,
# 1327, 1369, 1375, 1395, 1444, 1571, 1616, 1731, 1808, 1838,
# 1880, 2009, 2037, 2069, 2071, 2088, 2142, 2236, 2269, 2300,
# 2328, 2412, 2455, 2481, 2608, 2617, 2636, 2642, 2699, 2835,
# 2987, 3079, 3306, 3368, 3415, 3476, 3517, 3636, 3649, 3838,
# 3851, 3977, 4029, 4224, 4244, 4421, 4658, 4692, 4698, 4765,
# 4886, 4972, 5117, 5143, 5321, 5412, 5429, 5483, 5546, 5586,
# 5656, 5726, 5761, 5889, 5966, 6008, 6030, 6075, 6125, 6135,
# 6214, 6268, 6303, 6442, 6452, 6455, 6550, 6658, 6911, 6936,
# 6993, 7036, 7263, 7270, 7282, 7302, 7340, 7353, 7504, 7520,
# 7536, 7545, 7546, 7549, 7551, 7569, 7573, 7578, 7680, 7771,
# 7804, 7834, 7891, 8035, 8261, 8314, 8326, 8335, 8377, 8460,
# 8499, 8509, 8534, 8559, 8636, 8644, 8658, 8696, 8697, 8798,
# 9000, 9022, 9264, 9297, 9501, 9550, 9659, 9795, 9800, 9883
# ]

# 加载本地数据集
data_dir = r"D:\Fight-Poison-With-Poison\Cifar10-blend\test\data"
poisoned_dataset = PoisonedDataset(data_dir=data_dir, indices=indices, transform=transform)

# 创建数据加载器
poisoned_loader = DataLoader(poisoned_dataset, batch_size=1, shuffle=False, num_workers=0)  # 将 num_workers 设置为 0

# 初始化 VGG16 模型
model = vgg16_bn(num_classes=10)  # 初始化 ResNet-18 模型

# 加载保存的模型权重
model_save_path = r"D:\KSEM\VGG\clean_model\blend1_43k"
model.load_state_dict(torch.load(model_save_path, weights_only=True))  # 使用 weights_only=True 避免安全警告

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 统计预测为 0 的数量和比例
model.eval()
class_zero_count = 0  # 统计模型预测为 0 的数量
total_images = len(indices)

for idx, image in enumerate(poisoned_loader):
    image = image.to(device)

    # 模型预测
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)

    # 输出分类结果
    print(f"Index: {indices[idx]}, Predicted Class: {predicted.item()}")

    # 统计模型预测为 0 的数量
    if predicted.item() == 0:
        class_zero_count += 1

# 计算模型预测为 0 的占比
class_zero_ratio = class_zero_count / total_images * 100
print(f"Number of Class 0 Predictions: {class_zero_count}")
print(f"ASR: {class_zero_ratio:.2f}%")