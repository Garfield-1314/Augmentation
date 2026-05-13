# 数据增强与格式转换工具库

计算机视觉数据处理工具集合，包含图像增强、掩膜处理、数据集划分和格式转换功能。

## 环境要求

- **Python 版本**: 3.11.13
- **依赖包**:
  - opencv-python - 图像处理
  - matplotlib - 可视化
  - PyQt5 - 图形用户可视化界面
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
| **yolo_Au.py** | YOLO数据集专用增强管道（维护中...） |

### 可视化工具 (GUI)

- **`qt/main_gui.py`**: 提供强大的基于组件拖拽式图像处理与流水线编排桌面端界面软件。支持灵活设置输入输出、效果实时预览，以及无刷新状态非阻塞执行。运行方式：`python qt/main_gui.py`。

#### 打包为可执行程序 (.exe)
如果希望脱离 Python 环境运行，可以使用 PyInstaller 将 GUI 界面及其依赖打包为绿色免安装桌面程序：
```bash
# 激活环境并安装 pyinstaller
conda activate Au
pip install pyinstaller

# 执行打包命令
pyinstaller -w --name "Augmentation_GUI" --paths . --add-data "modules;modules" --add-data "example_images;example_images" --clean -y qt/main_gui.py
```
打包完成后，产物位于 `dist/Augmentation_GUI` 目录。将此整个文件夹发给其他用户，对方直接双击 `Augmentation_GUI.exe` 即可使用。

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
- 局部光照模拟（Local Illumination）：模拟自然光照产生的局部过曝（强光）与局部阴影（过暗）
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

### 模块化调用 (推荐)
自从 v2.0.0 版本起，该项目已全面模块化。你可以创建一个 `main.py` 来灵活组合功能：

```python
from Augmentation_AL import augment_dataset
import Augmentation_CV
import background

# 1. 生成纯净背景
background.batch_generate_backgrounds('../Datasets/bg', num_images=5)

# 2. 传统 CV 增强 (如压缩)
Augmentation_CV.batch_yasuo('../input', '../temp', w=160, h=160)

# 3. 高级 AL 增强 (生成 10 倍数据)
augment_dataset('../temp', '../output', num_augments=10)
```

或直接运行 `main.py` 查看演示：
```bash
python main.py
```

### 基础增强 (旧模式兼容)
```bash
# 依然可以直接运行脚本，内部包含默认配置
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
