# Data Augmentation & Format Conversion Toolkit

A comprehensive computer vision data processing toolkit featuring image augmentation, mask processing, dataset partitioning, and format conversion capabilities.

## Requirements

- **Python Version**: 3.11.13
- **Dependencies**:
  - opencv-python - Image processing
  - matplotlib - Data visualization
  - albumentationsx - Advanced data augmentation
  - tqdm - Progress bar display

## Installation

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Project Structure

### Core Modules

| File | Description |
|------|-------------|
| **Augmentation_AL.py** | Albumentations-based image augmentation (recommended) |
| **Augmentation_CV.py** | Traditional OpenCV enhancement methods (flip, rotate, noise, etc.) |
| **image_mask.py** | Background compositing and image mask processing |
| **image_mask_AL.py** | Albumentations-based mask augmentation |
| **yolo_Au.py** | YOLO dataset-specific augmentation pipeline |
| **shift_detection.py** | YOLO dataset partitioning (train/validation split) |
| **shift_classification.py** | Classification dataset processing |
| **background.py** | Background image management |

### Utility Tools (`another/` directory)

| File | Description |
|------|-------------|
| **xml2voc.py** | Convert XML annotations to VOC2007 format |
| **yolo2voc.py** | Convert YOLO format annotations to VOC format |
| **clean.py** | Dataset cleaning tool |

## Core Features

### 1. Image Augmentation
- Elastic distortion, optical distortion, rotation
- RGB color shifting, brightness/contrast adjustment
- Hue/saturation adjustment, motion blur

### 2. Image Masking
- Foreground object and background compositing
- Support for random scaling and position adjustment
- Visibility verification

### 3. Dataset Processing
- Automatic YOLO dataset partitioning
- Support for multiple annotation format conversions
- Batch dataset cleaning

## Usage Examples

### Basic Augmentation
```python
# Perform augmentation using Albumentations
python Augmentation_AL.py
```

### Format Conversion
```bash
# Convert XML to VOC format
python another/xml2voc.py --input_dir data --output_dir VOCdevkit

# Convert YOLO to VOC format
python another/yolo2voc.py
```

### Dataset Partitioning
```python
# Partition YOLO format dataset
python shift_detection.py
```

## Configuration Parameters

Key parameters in each script:
- `input_dir` - Input data directory
- `output_dir` - Output result directory
- `num_augments` - Number of augmented versions per image
- `train_ratio` - Training set ratio (default: 0.8)
- `random_seed` - Random seed (ensures reproducibility)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **Prepare your data**:
   - Place images in `input_dir`
   - Place annotations in the appropriate format

3. **Run augmentation**:
   ```bash
   python Augmentation_AL.py
   ```

4. **Check output**:
   - Augmented images will be saved to `output_dir`

## License

This project is provided as-is for research and development purposes.

## Support

For issues or questions, please refer to the documentation in each module's docstring.
