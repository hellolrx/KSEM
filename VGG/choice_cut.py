import torch
import torch.nn.utils.prune as prune
from torchvision import transforms
from utils.vgg import vgg16_bn

def selective_prune(model, prune_rate, num_layers_to_prune):
    """
    选择性剪枝函数。

    参数:
    model -- 要剪枝的模型
    prune_rate -- 剪枝的比例，例如0.2表示剪掉20%的权重
    num_layers_to_prune -- 要剪枝的层数（从后往前数）
    """
    # 获取所有卷积层和全连接层
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            layers.append((name, module))

    # 只对最后 num_layers_to_prune 层进行剪枝
    for name, module in layers[-num_layers_to_prune:]:
        # 应用随机剪枝
        prune.random_unstructured(module, name='weight', amount=prune_rate)
        # 剪枝后需要将模型转换为普通模块，否则剪枝不会生效
        prune.remove(module, 'weight')

# 载入VGG16_bn模型
model = vgg16_bn(num_classes=10)

# 设置数据转换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

# 加载原始模型
original_model_path = r"D:\KSEM\VGG\poision_model\cifar10_blend_seed26379_1.pth"
model.load_state_dict(torch.load(original_model_path, map_location=torch.device('cpu')))  # 加载权重
model.eval()  # 设置为评估模式

prune_rate = 0.60
num_layers_to_prune = 3
selective_prune(model, prune_rate, num_layers_to_prune)

# 打印剪枝后的模型参数数量，可以与剪枝前进行对比
print("剪枝后的模型参数数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 保存剪枝后的模型
save_path = r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed3.pth"
torch.save(model.state_dict(), save_path)
print(f"剪枝后的模型已保存到：{save_path}")


