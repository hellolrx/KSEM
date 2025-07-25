#15行更换毒化数据，46，47行更换中毒数据，131行更换模型保存路径 仅三处更改完成blend和badnet的切换
import os
import sys
import time
from tqdm import tqdm
from torchvision import transforms
from torch import nn
import torch
from utils import supervisor, tools
import config
from PIL import Image

# 固定数据集和参数
dataset = 'cifar10'
poison_type = 'blend'#'blend'
seed = 1234

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据变换
data_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# CIFAR-10 固定参数
num_classes = 10
arch = config.arch[dataset]
momentum = 0.9
weight_decay = 1e-4
epochs = 85             #100
milestones = torch.tensor([50, 75])
learning_rate = 0.1
batch_size = 128
kwargs = {'num_workers': 0, 'pin_memory': True}

# 指定训练数据和标签路径
poisoned_set_img_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\35k_1k\image"
poisoned_set_label_path = r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\train_on_clean_set\labels\labels.pth"

# 自定义数据集类，按文件顺序加载
class FileOrderIMG_Dataset(tools.IMG_Dataset):
    def __init__(self, data_dir, label_path, transforms=None):
        super().__init__(data_dir, label_path, transforms)
        # 获取文件夹中的文件列表，按文件名排序
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

    def __getitem__(self, index):
        # 按文件顺序加载
        filename = self.files[index]
        img_path = os.path.join(self.dir, filename)
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        # 按顺序从 labels.pth 中取标签
        label = self.gt[index]
        return img, label  # 不返回 filename，避免 DataLoader 报错

    def __len__(self):
        return len(self.files)  # 返回文件数量

# 加载毒化数据集
print(f"Loading training dataset from: {poisoned_set_img_dir}")
try:
    poisoned_set = FileOrderIMG_Dataset(
        data_dir=poisoned_set_img_dir,
        label_path=poisoned_set_label_path,
        transforms=data_transform_aug
    )
    #print(poisoned_set[25])
    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=tools.worker_init,
        **kwargs
    )
except Exception as e:
    print(f"Error loading poisoned dataset: {e}")
    sys.exit(1)


test_set_img_dir = r"D:\KSEM\clean_set\cifar10\test_split\data"
test_set_label_path = r"D:\KSEM\clean_set\cifar10\test_split\labels"
print(f"Loading test dataset from: {test_set_img_dir}")
try:
    test_set = tools.IMG_Dataset(
        data_dir=test_set_img_dir,
        label_path=test_set_label_path,
        transforms=data_transform
    )
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=tools.worker_init,
        **kwargs
    )
except Exception as e:
    print(f"Error loading test dataset: {e}")
    sys.exit(1)

# 毒化变换（用于测试）
trigger_dir = r"D:\KSEM\VGG\triggers"
trigger_name = config.trigger_default[poison_type]
trigger_path = os.path.join(trigger_dir, trigger_name)

poison_transform = supervisor.get_poison_transform(
    poison_type=poison_type,
    dataset_name=dataset,
    target_class=config.target_class[dataset],
    trigger_transform=data_transform,
    is_normalized_input=True,
    alpha=0.2,
    trigger_name=trigger_path,
)

# 初始化模型
model = arch(num_classes=num_classes)
model = nn.DataParallel(model)
model = model.to(device)

# 模型保存路径
model_save_dir = r"D:\KSEM\VGG\blend_diff_poison_rate\21457cifar10_1blendpoisoned\poison_model\36k"
os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
print(f"Will save to '{model_save_dir}'.")
if os.path.exists(model_save_dir):
    print(f"Model '{model_save_dir}' already exists!")

# 定义损失函数、优化器和调度器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones.tolist())

# 训练参数
source_classes = None
all_to_all = False

# 训练循环
st = time.time()
for epoch in range(1, epochs + 1):
    start_time = time.perf_counter()

    # Train
    model.train()
    try:
        for data, target in tqdm(poisoned_set_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
    except Exception as e:
        print(f"Error during training epoch {epoch}: {e}")
        sys.exit(1)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(
        f'<Backdoor Training> Train Epoch: {epoch} \tLoss: {loss.item():.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}, Time: {elapsed_time:.2f}s')
    scheduler.step()

    # Test
    tools.test(
        model=model,
        test_loader=test_set_loader,
        poison_test=True,
        poison_transform=poison_transform,
        num_classes=num_classes,
        source_classes=source_classes,
        all_to_all=all_to_all
    )
    torch.save(model.module.state_dict(), model_save_dir)
    print("")

# 最终保存模型
torch.save(model.module.state_dict(), model_save_dir)