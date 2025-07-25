import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils.tools import IMG_Dataset

from utils import resnet
from tqdm import tqdm
import shutup

shutup.please()

# 定义数据变换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 创建毒化测试数据加载器

# poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\Cifar10_TaCT_22\train\data",
#                            label_path=r"D:\Fight-Poison-With-Poison\Cifar10_TaCT_22\train\labels",
#                            transforms=data_transform)

#blend
# poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\blend_poision\data",
#                             label_path=r"D:\Fight-Poison-With-Poison\blend_poision\labels",
#                             transforms=data_transform)

#badnet
poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\Badnet_poision\data",
                           label_path=r"D:\Fight-Poison-With-Poison\Badnet_poision\labels",
                           transforms=data_transform)

# poisoned_set = IMG_Dataset(data_dir=r"D:\Fight-Poison-With-Poison\Trojan_poision\data",
#                            label_path=r"D:\Fight-Poison-With-Poison\Trojan_poision\labels",
#                            transforms=data_transform)

# 加载模型
model_path = r"D:\KSEM\ResNet\cutted_model\Badnet_10_40_seed0.pth"
model = resnet.ResNet18(num_classes=10)

# 自动选择设备，如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到选择的设备
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 将模型设置为评估模式

# 初始化变量
total_images = len(poisoned_set)
correct_classification_all = 0
correct_classification_indices = 0

# 指定的索引数组

#badnet
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

#TaCT
# indices =[
#     238, 833, 840, 962, 1781, 1790, 2038, 2771, 2958, 3327,
#     5112, 5197, 5236, 5261, 5459, 5663, 5841, 6074, 6701, 6709,
#     6798, 7019, 7255, 7349, 7600, 7605, 7821, 7832, 8107, 8848,
#     9007, 9012, 9317, 9476, 9717, 11543, 11782, 12158, 13010, 13032,
#     13052, 13508, 13509, 13523, 14713, 15577, 15770, 15779, 15883, 16029,
#     16800, 17275, 17460, 17586, 17746, 17752, 18247, 18311, 18425, 18583,
#     19046, 19341, 19481, 20114, 21283, 21752, 21984, 22269, 23489, 23548,
#     23929, 24110, 24157, 24398, 25718, 26065, 26241, 26344, 26550, 26737,
#     27130, 28086, 28530, 28605, 28848, 29121, 29146, 29176, 29308, 29394,
#     29546, 30103, 30313, 30545, 30673, 30747, 30865, 31126, 31761, 31969,
#     32624, 32684, 33059, 33305, 34063, 34090, 34206, 34420, 34548, 34778,
#     34850, 35174, 36618, 37327, 37455, 37494, 37596, 37937, 38175, 38270,
#     38929, 39256, 39349, 39422, 39769, 39805, 40615, 40759, 41061, 41445,
#     41502, 41911, 42753, 42925, 43232, 43825, 44415, 44620, 45187, 45399,
#     46003, 46730, 47035, 47568, 47749, 48898, 49066, 49426, 49702, 49733
# ]

#blend
# indices = [
#     231, 422, 473, 547, 1034, 2331, 2936, 3363, 3871, 4113, 4205, 4709, 5348, 5431, 5473, 5629, 5833, 6430, 6471, 6723,
#     7363, 7887, 8095, 8308, 8595, 9172, 9466, 9740, 10453, 10960, 11131, 11936, 12545, 12581, 13285, 13338, 13616, 14011,
#     14105, 14598, 14656, 14772, 14818, 15238, 15245, 15307, 15775, 15867, 15987, 16124, 16221, 17249, 17413, 17569, 17784,
#     17818, 18084, 18343, 19355, 20178, 20356, 21661, 22019, 22515, 22572, 22600, 23169, 23357, 23631, 23900, 24469, 24723,
#     25060, 25405, 25794, 26190, 26295, 26367, 26582, 26804, 26862, 27750, 28264, 28559, 28734, 29014, 29063, 29433, 30055,
#     30536, 30637, 30733, 31245, 31662, 32151, 32793, 33178, 33265, 33782, 34306, 34572, 34731, 34813, 34817, 35097, 36099,
#     36101, 36424, 36806, 36983, 37288, 37895, 38472, 38886, 38945, 39097, 39214, 39463, 39663, 40041, 40937, 41272, 41412,
#     41660, 41667, 42008, 42113, 42164, 42403, 42860, 43560, 43736, 44135, 44446, 44600, 45160, 45464, 45470, 45486, 45906,
#     46712, 47051, 47448, 48367, 48893, 49277, 49516, 49902, 49950, 49961
# ]

#trojan
# indices = [
#     120, 236, 595, 734, 1104, 1190, 1838, 2379, 2641, 2711,
#     2807, 2990, 3324, 3635, 3775, 4562, 5118, 5198, 5551, 6519,
#     7112, 8690, 8747, 9168, 9179, 9189, 9269, 10155, 10345, 10557,
#     10871, 11083, 11357, 11771, 11976, 12142, 12259, 12376, 12455, 12956,
#     13469, 13659, 13727, 13932, 14437, 14461, 14554, 15200, 15574, 15925,
#     15974, 16156, 17469, 17493, 17564, 17650, 17731, 18031, 18318, 18328,
#     18653, 18959, 19140, 19484, 19679, 19783, 19891, 20227, 20464, 20800,
#     22911, 23064, 23277, 23504, 23533, 23693, 23837, 23905, 24564, 24738,
#     25774, 26061, 26841, 26884, 27273, 28191, 28250, 29166, 30143, 30356,
#     30724, 30902, 31244, 31499, 31717, 32311, 32672, 33307, 33705, 33859,
#     34638, 34866, 34871, 34881, 35766, 36934, 38395, 38514, 38722, 38875,
#     39035, 39072, 39419, 40042, 40148, 40273, 40390, 40530, 40627, 41059,
#     41302, 41945, 42136, 42168, 42207, 42648, 42748, 42914, 43296, 43491,
#     43915, 43977, 44194, 44251, 44266, 45320, 45327, 45724, 45925, 46733,
#     46951, 46972, 47722, 48286, 48669, 48860, 48954, 49115, 49356, 49840
# ]

# 使用DataLoader按批次加载数据，设置batch_size为128
data_loader_all = DataLoader(poisoned_set, batch_size=128, shuffle=False)

# 使用tqdm创建进度条，用于遍历整个数据集
pbar_all = tqdm(total=total_images, desc="Classifying all images and checking consistency")

# 遍历整个数据集（按批次）
with torch.no_grad():
    for batch_idx, (batch_images, batch_labels) in enumerate(data_loader_all):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_images)
        _, predicted = torch.max(outputs, 1)

        # 统计当前批次中正确分类的样本数量，并更新总的正确分类数量
        correct_num_in_batch = (predicted == batch_labels).sum().item()
        correct_classification_all += correct_num_in_batch

        # 更新进度条
        pbar_all.update(len(batch_images))

pbar_all.close()

# 创建一个新的DataLoader，用于按批次加载指定索引对应的数据
subset_indices = torch.tensor(indices).long()
subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
data_loader_indices = DataLoader(poisoned_set, batch_size=128, sampler=subset_sampler)

# 使用tqdm创建进度条，用于遍历指定的索引数组
pbar_indices = tqdm(total=len(indices), desc="Classifying specified indices and checking consistency")

# 遍历指定的索引数组（按批次）
with torch.no_grad():
    for batch_idx, (batch_images, batch_labels) in enumerate(data_loader_indices):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_images)
        _, predicted = torch.max(outputs, 1)

        # 统计当前批次中正确分类的样本数量，并更新指定索引对应的正确分类数量
        correct_num_in_batch = (predicted == batch_labels).sum().item()
        correct_classification_indices += correct_num_in_batch

        # 更新进度条
        pbar_indices.update(len(batch_images))

pbar_indices.close()

# 打印分类一致的图片数量和一致性百分比
print(f"Number of images with correct classification in all images: {correct_classification_all}")
print(f"Percentage of correct classification in all images: {correct_classification_all / total_images * 100:.2f}%")
print(f"Number of images with correct classification in specified indices: {correct_classification_indices}")
print(f"Percentage of correct classification in specified indices: {correct_classification_indices / len(indices) * 100:.2f}%")

# #首次验证新被剪枝模型ACC需要开启
# cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
# cifar10_testloader = DataLoader(cifar10_testset, batch_size=128, shuffle=False)
#
# # 初始化CIFAR-10验证的变量
# correct_classification_cifar10 = 0
# total_cifar10_images = len(cifar10_testset)
#
# # 使用tqdm创建进度条，用于遍历CIFAR-10数据集
# pbar_cifar10 = tqdm(total=total_cifar10_images, desc="Classifying CIFAR-10 images")
#
# # 遍历CIFAR-10数据集（按批次）
# with torch.no_grad():
#     for batch_idx, (batch_images, batch_labels) in enumerate(cifar10_testloader):
#         batch_images = batch_images.to(device)
#         batch_labels = batch_labels.to(device)
#         outputs = model(batch_images)
#         _, predicted = torch.max(outputs, 1)
#
#         # 统计当前批次中正确分类的样本数量，并更新总的正确分类数量
#         correct_num_in_batch = (predicted == batch_labels).sum().item()
#         correct_classification_cifar10 += correct_num_in_batch
#
#         # 更新进度条
#         pbar_cifar10.update(len(batch_images))
#
# pbar_cifar10.close()
#
# # 打印CIFAR-10验证结果
# print(f"Number of images with correct classification in CIFAR-10: {correct_classification_cifar10}")
# print(f"Percentage of correct classification in CIFAR-10: {correct_classification_cifar10 / total_cifar10_images * 100:.2f}%")