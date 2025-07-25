#数据合并，将抽取的数据和干净数据合并
import os
import shutil
import torch
from tqdm import tqdm

# 定义路径
input_dirs = [
    {
        "image_dir": r"D:\KSEM\ResNet\Trojan\38k\image",
        "label_path": r"D:\KSEM\ResNet\Trojan\38k\labels\labels.pth",
        "indices_path": r"D:\KSEM\ResNet\Trojan\38k\labels\indices.pth"
    },
    {
        "image_dir": r"D:\KSEM\ResNet\Trojan\2k_2k\clean_set\image",
        "label_path": r"D:\KSEM\ResNet\Trojan\2k_2k\clean_set\labels\labels.pth",
        "indices_path": r"D:\KSEM\ResNet\Trojan\2k_2k\clean_set\labels\indices.pth"
    },
    {
        "image_dir": r"D:\KSEM\ResNet\Trojan\2k_500\clean_set\image",
        "label_path": r"D:\KSEM\ResNet\Trojan\2k_500\clean_set\labels\labels.pth",
        "indices_path": r"D:\KSEM\ResNet\Trojan\2k_500\clean_set\labels\indices.pth"
    },
    {
        "image_dir": r"D:\KSEM\ResNet\Trojan\3k_2k\clean_set\image",
        "label_path": r"D:\KSEM\ResNet\Trojan\3k_2k\clean_set\labels\labels.pth",
        "indices_path": r"D:\KSEM\ResNet\Trojan\3k_2k\clean_set\labels\indices.pth"
    },
    # {
    #     "image_dir": r"D:\KSEM\ResNet\TaCT\22k_18k\clean_set\image",
    #     "label_path": r"D:\KSEM\ResNet\TaCT\22k_18k\clean_set\labels\labels.pth",
    #     "indices_path": r"D:\KSEM\ResNet\TaCT\22k_18k\clean_set\labels\indices.pth"
    # },

    # {
    #     "image_dir": r"D:\KSEM\ResNet\TaCT\4k_4k\clean_set\image",
    #     "label_path": r"D:\KSEM\ResNet\TaCT\4k_4k\clean_set\labels\labels.pth",
    #     "indices_path": r"D:\KSEM\ResNet\TaCT\4k_4k\clean_set\labels\indices.pth"
    # },
    # {
    #     "image_dir": r"D:\KSEM\ResNet\Trojan\extract\image",
    #     "label_path": r"D:\KSEM\ResNet\Trojan\extract\labels\labels.pth",
    #     "indices_path": r"D:\KSEM\ResNet\Trojan\extract\labels\indices.pth"
    # }

]

output_image_dir = r"D:\KSEM\ResNet\Trojan\43k\image"
output_label_dir = r"D:\KSEM\ResNet\Trojan\43k\labels"
output_label_path = os.path.join(output_label_dir, "labels.pth")
output_indices_path = os.path.join(output_label_dir, "indices.pth")

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 初始化用于合并的字典，以索引为键，存储文件名和标签
index_to_data = {}

# 遍历每个输入目录
for input_set in input_dirs:
    image_dir = input_set["image_dir"]
    label_path = input_set["label_path"]
    indices_path = input_set["indices_path"]

    # 加载标签和索引
    labels = torch.load(label_path, weights_only=False)
    indices = torch.load(indices_path, weights_only=False)
    print(f"Loaded {len(labels)} labels from {label_path}")
    print(f"Loaded {len(indices)} indices from {indices_path}")

    # 获取图片文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    print(f"Found {len(image_files)} images in {image_dir}, First 10: {image_files[:10]}")

    # 验证数量一致性
    if len(image_files) != len(labels) or len(image_files) != len(indices):
        raise ValueError(f"Inconsistent counts in {image_dir}: images={len(image_files)}, labels={len(labels)}, indices={len(indices)}")

    # 构建索引到文件名和标签的映射
    for i, image_file in enumerate(image_files):
        idx = indices[i].item()
        idx_from_filename = int(image_file.split('.')[0])
        if idx != idx_from_filename:
            print(f"Warning: Index {idx} does not match filename {image_file} in {indices_path}")
        # 存储文件名和标签，以索引为键
        index_to_data[idx] = {
            "image_file": image_file,
            "label": labels[i].item(),
            "source_dir": image_dir
        }

# 按索引升序排序并合并
sorted_indices = sorted(index_to_data.keys())
combined_labels = []
combined_indices = []

for idx in tqdm(sorted_indices, desc="Merging datasets"):
    data = index_to_data[idx]
    image_file = data["image_file"]
    label = data["label"]
    src_image_path = os.path.join(data["source_dir"], image_file)
    dst_image_path = os.path.join(output_image_dir, image_file)

    # 复制图片（如果目标文件已存在则覆盖）
    shutil.copy2(src_image_path, dst_image_path)

    # 添加标签和索引
    combined_labels.append(label)
    combined_indices.append(idx)

# 将合并后的标签和索引转换为 torch.LongTensor 并保存
combined_labels_tensor = torch.LongTensor(combined_labels)
combined_indices_tensor = torch.LongTensor(combined_indices)

torch.save(combined_labels_tensor, output_label_path)
torch.save(combined_indices_tensor, output_indices_path)

# 打印保存信息
print(f"\nCombined dataset saved to {output_image_dir}, labels: {output_label_path}, indices: {output_indices_path}")

# 验证数量是否匹配
img_count = len(os.listdir(output_image_dir))
label_count = len(combined_labels)
indices_count = len(combined_indices)

print(f"\nCombined dataset: {img_count} images, {label_count} labels, {indices_count} indices")
if img_count == label_count == indices_count:
    print("Data, labels, and indices match in number!")
else:
    print("Warning: Data, labels, or indices do not match in number!")

# 验证合并后数据和标签的匹配性
combined_image_files = sorted([f for f in os.listdir(output_image_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
print(f"Validating combined dataset with {len(combined_image_files)} images")
mismatch_count = 0
for i, image_file in enumerate(combined_image_files):
    idx_from_filename = int(image_file.split('.')[0])
    if idx_from_filename != combined_indices[i].item():
        print(f"Mismatch: Filename {image_file} does not match index {combined_indices[i].item()}")
        mismatch_count += 1
    if idx_from_filename in index_to_data and index_to_data[idx_from_filename]["label"] != combined_labels[i].item():
        print(f"Mismatch: Filename {image_file} label {index_to_data[idx_from_filename]['label']} does not match combined label {combined_labels[i].item()}")
        mismatch_count += 1

if mismatch_count == 0:
    print("Validation passed: All data and labels match correctly!")
else:
    print(f"Validation failed: Found {mismatch_count} mismatches in combined dataset!")