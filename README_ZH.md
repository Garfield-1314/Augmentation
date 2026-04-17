# 数据增强与格式转换工具库

计算机视觉数据处理工具集合，包含图像增强、掩膜处理、数据集划分和格式转换功能。

## 环境要求

- **Python 版本**: 3.11.13
- **依赖包**:
  - opencv-python - 图像处理
  - matplotlib - 可视化
  - albumentationsx - 高级数据增强
  - tqdm - 进度条显示

## 安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 项目结构

### 核心模块

| 文件 | 功能说明 |
|------|--------|
| **Augmentation_AL.py** | 基于Albumentations的图像增强（推荐使用） |
| **Augmentation_CV.py** | 基于OpenCV的传统增强方法（翻转、旋转、噪声等） |
| **image_mask.py** | 背景合成与图像掩膜处理 |
| **image_mask_AL.py** | 基于Albumentations的掩膜增强 |
| **yolo_Au.py** | YOLO数据集专用增强管道 |
| **shift_detection.py** | YOLO数据集划分（训练/验证集分割） |
| **shift_classification.py** | 分类数据集处理 |
| **background.py** | 背景图像管理 |

### 辅助工具 (`another/` 目录)

| 文件 | 功能说明 |
|------|--------|
| **xml2voc.py** | 将XML标注转换为VOC2007格式 |
| **yolo2voc.py** | 将YOLO格式标注转换为VOC格式 |
| **clean.py** | 数据集清理工具 |

## 核心功能

### 1. 图像增强
- 弹性变形、光学畸变、旋转
- RGB色彩变换、亮度对比度调整
- 色调/饱和度调整、运动模糊

### 2. 图像掩膜
- 前景物体与背景合成
- 支持随机缩放与位置调整
- 可见性检验

### 3. 数据集处理
- YOLO数据集自动划分
- 支持多种标注格式转换
- 批量数据清理

## 使用示例

### 基础增强
```python
# 使用Albumentations进行增强
python Augmentation_AL.py
```

### 格式转换
```bash
# XML转VOC格式
python another/xml2voc.py --input_dir data --output_dir VOCdevkit

# YOLO转VOC格式
python another/yolo2voc.py
```

### 数据集划分
```python
# 划分YOLO格式数据集
python shift_detection.py
```

## 配置说明

各脚本中的关键参数：
- `input_dir` - 输入数据目录
- `output_dir` - 输出结果目录
- `num_augments` - 每张图片生成的增强版本数
- `train_ratio` - 训练集比例（默认0.8）
- `random_seed` - 随机种子（确保可重复性）

## 快速开始

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **准备数据**:
   - 将图像放入 `input_dir` 目录
   - 将标注放入对应格式目录

3. **运行增强**:
   ```bash
   python Augmentation_AL.py
   ```

4. **查看结果**:
   - 增强后的图像将保存到 `output_dir` 目录

## 许可证

本项目仅供研究和开发使用。

## 联系方式

如有问题或建议，请参考各模块中的文档说明。
