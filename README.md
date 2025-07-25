# 🎉 KSEM：AI安全与模型实验乐园！

欢迎来到 KSEM 项目！  
这里是你探索AI安全、模型鲁棒性、数据投毒与防御的绝佳试验田！  
**澳门城市大学**出品，科研&工程两开花，代码注释友好，结构清晰，欢迎 Star & Fork！

---

## 🗂️ 项目结构

```
KSEM/
├── poison_tool_box/   # 各类数据投毒与防御算法
├── ResNet/            # ResNet相关实验与工具
├── VGG/               # VGG相关实验与工具
```

---

## 🚦 总体流程

1. 中毒数据生成
2. 中毒模型训练
3. 中毒模型被剪枝
4. 剪枝模型分类正确取交集
5. 索引加载类别并补充数据
6. 干净数据聚类并训练模型
7. 利用干净模型使中毒数据聚类
8. 中毒数据训练并二次干净数据聚类
9. 多批干净数据合成并训练

---

## 🧩 主要脚本及用法说明

### 1. 数据处理与分割

- **extract_class.py**  
  用于从指定类别中随机抽取部分样本，生成新的数据集。  
  **用法**：直接运行，修改脚本内路径和类别参数即可。

- **filter_poision_set.py**  
  按索引分割数据集，将数据分为干净集和含毒集，并保存对应标签。  
  **用法**：直接运行，确保 index_file、数据路径等参数正确。

---

### 2. 模型训练与验证

- **train_clean_model.py**  
  训练干净数据的模型。支持自定义数据集路径、标签、批量大小等。  
  **用法**：直接运行，修改 img_dir 和 labels_file 路径为你的干净数据。

- **train_on_poisoned_set.py**  
  训练中毒数据的模型。支持多种投毒方式（如trojan、blend、badnet），只需更改 poison_type 和数据路径。  
  **用法**：直接运行，按注释修改数据路径和 poison_type。

---

### 3. 模型评估

- **vaild_clean_model.py**  
  验证干净模型在CIFAR-10测试集上的准确率。  
  **用法**：直接运行，修改 model_save_path 为你的模型权重路径。

- **vaild_poision_model.py**  
  验证中毒模型在特定中毒数据集上的表现，支持多种攻击类型。  
  **用法**：直接运行，按注释修改数据路径和模型路径。

---

### 4. 其他常用脚本

- **combin_data.py**  
  用于合并不同来源的数据集，生成新的训练集。
- **count_match_trigger.py**  
  统计触发器命中情况，分析后门攻击效果。
- **cross_vaild.py**  
  跨验证脚本，支持多折交叉实验。
- **sort_loss.py / filter_loss.py / confidence_filter.py**  
  各类损失分析与过滤脚本，助力模型鲁棒性提升。

---

## 🦾 VGG 相关脚本

VGG 文件夹下脚本与 ResNet 逻辑一致，只需将模型和路径替换为 VGG 相关即可。

---

## 🏁 快速开始

1. 克隆本仓库：
   ```bash
   git clone https://github.com/hellolrx/KSEM.git
   ```
2. 进入对应文件夹，选择你感兴趣的脚本运行即可！

---

## 💬 交流与贡献

- 欢迎 issue、PR、star！
- 有任何想法、建议、bug，尽管提出来，我们一起让 KSEM 更加酷炫！

---

> **作者：Rixi Liang¹, Shuai Zhou¹*, Mingxu Zhu¹, Chi Liu¹, Minfeng Qi¹*  
> *通讯作者：Shuai Zhou  
> City University of Macau 计算机科学硕士在读  
> [GitHub主页](https://github.com/hellolrx)
