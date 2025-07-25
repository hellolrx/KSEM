import torch
import time
from torchvision import transforms
from utils.tools import IMG_Dataset
from tqdm import tqdm
import os
import random
from utils import resnet
from torchvision.utils import save_image
from PIL import Image


def set_random_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def classify_and_record_correct_indices(model_path, poisoned_set, device, seed_value):
    set_random_seed(seed_value)  # 设置随机种子
    model = resnet.ResNet18(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct_indices = []

    with torch.no_grad():
        for idx in tqdm(range(len(poisoned_set)), desc=f"Classifying with {model_path}"):
            try:
                images, labels = poisoned_set[idx]
                if images is None or labels is None:  # 跳过无效数据
                    continue
                images = images.to(device)
                labels = labels.to(device)
                images = images.unsqueeze(0)  # 添加批次维度
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                if predicted[0].item() == labels.item():
                    correct_indices.append(idx)
            except FileNotFoundError:
                print(f"File not found for index {idx}. Skipping...")
                continue
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

    num_correct = len(correct_indices)
    print(f"Model {model_path} correctly classified {num_correct} images.")

    return correct_indices  # 只返回正确索引，不记录分类时间


def save_processed_dataset(correct_indices, poisoned_set, output_data_dir, output_label_path):
    """
    只保存 correct_indices 为 indices.pth 文件，不生成图片和标签文件。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_label_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将 correct_indices 转换为 torch.LongTensor 并保存为 .pth 文件
    indices_tensor = torch.LongTensor(correct_indices)
    indices_path = os.path.join(output_dir, "indices.pth")
    torch.save(indices_tensor, indices_path)

    print(f"Saved {len(correct_indices)} indices to {indices_path}.")


def main():
    # 记录程序总开始时间
    start_time_total = time.time()

    # 定义数据变换，用于模型输入（标准化）
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    # 加载原始毒化数据集（Blend_poision）
    poisoned_set = IMG_Dataset(
        data_dir=r"D:\Fight-Poison-With-Poison\Trojan_poision\data",
        label_path=r"D:\Fight-Poison-With-Poison\Trojan_poision\labels",
        transforms=data_transform
    )

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径列表
    model_paths = [
        r"D:\KSEM\ResNet\cutted_model\Trojan_6_65_seed0.pth",
        r"D:\KSEM\ResNet\cutted_model\Trojan_6_65_seed1.pth",
        r"D:\KSEM\ResNet\cutted_model\Trojan_6_65_seed2.pth",
        r"D:\KSEM\ResNet\cutted_model\Trojan_6_65_seed3.pth",
        r"D:\KSEM\ResNet\cutted_model\Trojan_6_65_seed4.pth"
    ]

    correct_indices_all_models = []

    # 记录所有模型分类的时间
    for i, model_path in enumerate(model_paths):
        correct_indices = classify_and_record_correct_indices(model_path, poisoned_set, device, i)
        correct_indices_all_models.append(correct_indices)

    # 找出所有模型都正确分类的索引
    all_correct_indices = sorted(set.intersection(*map(set, correct_indices_all_models)))

    num_all_correct = len(all_correct_indices)
    print(f"Number of images correctly classified by all models: {num_all_correct}")

    # 定义输出路径（保存处理后的数据集）
    global output_dir  # 设为全局变量，以便在函数中使用
    output_dir = r"D:\KSEM\ResNet\Trojan\all_correct_combin"
    output_data_dir = os.path.join(output_dir, "data")
    output_label_path = os.path.join(output_dir, "labels")  # 原始路径不带扩展名

    # 保存处理后的数据集（只保存 indices）
    save_processed_dataset(all_correct_indices, poisoned_set, output_data_dir, output_label_path)

    # 记录程序总结束时间
    end_time_total = time.time()
    total_execution_time = end_time_total - start_time_total
    print(f"Total execution time: {total_execution_time:.2f} seconds")


if __name__ == '__main__':
    main()
