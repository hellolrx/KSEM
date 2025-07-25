import torch
from torchvision import transforms
from utils.tools import IMG_Dataset  # 假设你有这个模块
from tqdm import tqdm
import os
import random
from utils.vgg import vgg16_bn
from torchvision.utils import save_image
from PIL import Image

def set_random_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def classify_and_record_correct_indices(model_path, poisoned_set, device, seed_value):
    set_random_seed(seed_value)  # 设置随机种子
    model = vgg16_bn(num_classes=10)
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
    return correct_indices

def save_processed_dataset(correct_indices, poisoned_set, output_data_dir, output_label_path):
    """
    将处理后的数据和标签保存为与训练代码兼容的格式：
    - 图像保存为 PNG 文件，保留原始文件名（例如 86.png）
    - 标签保存为 .pth 文件，按 correct_indices 顺序存储
    - 索引保存为 .pth 文件，与图片文件名对应
    """
    # 如果输出文件夹存在，直接覆盖
    if os.path.exists(output_data_dir):
        for file in os.listdir(output_data_dir):
            os.remove(os.path.join(output_data_dir, file))
    else:
        os.makedirs(output_data_dir)

    # 准备保存的标签和索引列表
    label_set = []
    indices_set = []

    # 定义仅转换为张量的变换，不标准化
    raw_transform = transforms.ToTensor()

    # 保存图像、标签和索引，保留原始文件名
    for orig_idx in tqdm(correct_indices, desc="Saving processed dataset", total=len(correct_indices)):
        try:
            # 从原始路径加载图像，保持原始颜色
            orig_img_path = os.path.join(poisoned_set.dir, f"{orig_idx}.png")
            image = Image.open(orig_img_path).convert("RGB")
            image_tensor = raw_transform(image)  # 仅转换为张量，不标准化

            # 保存图像，保留原始文件名
            img_file_name = f"{orig_idx}.png"
            img_file_path = os.path.join(output_data_dir, img_file_name)
            save_image(image_tensor, img_file_path)  # 保存原始颜色图像

            # 获取标签
            _, label = poisoned_set[orig_idx]
            label_set.append(label.item())
            indices_set.append(orig_idx)  # 记录原始索引，与图片文件名对应
        except Exception as e:
            print(f"Error saving original index {orig_idx}: {e}")
            continue

    # 将标签转换为 torch.LongTensor 并保存为 .pth 文件
    label_set_tensor = torch.LongTensor(label_set)
    output_label_pth_path = os.path.join(os.path.dirname(output_label_path), "labels.pth")  # 确保路径正确
    torch.save(label_set_tensor, output_label_pth_path)
    print(f"Saved {len(label_set)} processed labels to {output_label_pth_path}")

    # 将索引转换为 torch.LongTensor 并保存为 .pth 文件
    indices_set_tensor = torch.LongTensor(indices_set)
    output_indices_pth_path = os.path.join(os.path.dirname(output_label_path), "indices.pth")
    torch.save(indices_set_tensor, output_indices_pth_path)
    print(f"Saved {len(indices_set)} processed indices to {output_indices_pth_path}")

    # 保存 correct_indices 以便后续加载时使用
    indices_path = os.path.join(output_dir, "correct_indices.pth")
    torch.save(torch.LongTensor(correct_indices), indices_path)  # 确保保存为 tensor
    print(f"Saved correct indices to {indices_path}")

    # 验证数量是否匹配
    img_count = len(os.listdir(output_data_dir))
    label_count = len(label_set)
    indices_count = len(indices_set)
    print(f"\nDataset: {img_count} images, {label_count} labels, {indices_count} indices")
    if img_count == label_count == indices_count:
        print("Data, labels, and indices match in number!")
    else:
        print("Warning: Data, labels, or indices do not match in number!")

def main():
    # 定义数据变换，用于模型输入（标准化）
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    # 加载原始毒化数据集（Blend_poision）
    poisoned_set = IMG_Dataset(
        data_dir=r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\data",
        label_path=r"D:\KSEM\VGG\blend_diff_poison_rate\26379cifar10_1blendpoisoned\labels",
        transforms=data_transform
    )

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径列表
    model_paths = [
        r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed0.pth",
        r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed1.pth",
        r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed2.pth",
        r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed3.pth",
        r"D:\KSEM\VGG\cutted_model\blend3_55_1_seed4.pth"
    ]

    # 获取每个模型正确分类的索引
    correct_indices_all_models = [
        classify_and_record_correct_indices(model_path, poisoned_set, device, i)
        for i, model_path in enumerate(model_paths)
    ]

    # 找出所有模型都正确分类的索引
    all_correct_indices = sorted(set.intersection(*map(set, correct_indices_all_models)))  # 排序以保持一致性
    num_all_correct = len(all_correct_indices)
    print(f"Number of images correctly classified by all models: {num_all_correct}")

    # 定义输出路径（保存处理后的数据集）
    global output_dir  # 设为全局变量，以便在函数中使用
    output_dir = r"D:\KSEM\VGG\blend\0\1"
    output_data_dir = os.path.join(output_dir, "data")
    output_label_path = os.path.join(output_dir, "labels")  # 原始路径不带扩展名

    # 保存处理后的数据集
    save_processed_dataset(all_correct_indices, poisoned_set, output_data_dir, output_label_path)


if __name__ == '__main__':
    main()