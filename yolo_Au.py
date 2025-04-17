import os
import random
from PIL import Image, ImageEnhance
import shutil

# 配置参数
dataset_dir = 'nums'       # 原始数据集根目录
output_dir = 'AU'             # 增强数据输出目录
num_augments = 5              # 每张图片生成增强样本的数量
subsets = ['train', 'val']    # 需要处理的子集

# 增强参数范围
brightness_range = (0.6, 1.4)    # 亮度调整范围
contrast_range = (0.6, 1.4)      # 对比度调整范围
saturation_range = (0.5, 1.5)    # 饱和度调整范围
hue_range = (-0.25, 0.25)          # 色调调整范围

def process_dataset():
    # 创建输出目录结构
    for subset in subsets:
        aug_image_dir = os.path.join(output_dir, 'images', subset)
        aug_label_dir = os.path.join(output_dir, 'labels', subset)
        os.makedirs(aug_image_dir, exist_ok=True)
        os.makedirs(aug_label_dir, exist_ok=True)

    # 遍历所有子集（train/val）
    for subset in subsets:
        image_subdir = os.path.join(dataset_dir, 'images', subset)
        label_subdir = os.path.join(dataset_dir, 'labels', subset)
        
        # 遍历子集中的图片
        for image_name in os.listdir(image_subdir):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_subdir, image_name)
            label_path = os.path.join(label_subdir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                continue

            # 生成指定数量的增强样本
            for i in range(num_augments):
                augment_sample(image_path, label_path, subset, i)

def augment_sample(image_path, label_path, subset, aug_idx):
    # 加载原始数据
    image = Image.open(image_path).convert('RGB')
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # 应用颜色增强
    augmented_image = apply_color_augmentation(image)
    
    # 生成输出路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image = os.path.join(output_dir, 'images', subset, f"{base_name}_aug{aug_idx}.jpg")
    output_label = os.path.join(output_dir, 'labels', subset, f"{base_name}_aug{aug_idx}.txt")

    # 保存增强结果
    augmented_image.save(output_image)
    with open(output_label, 'w') as f:
        f.write('\n'.join(labels))

def apply_color_augmentation(image):
    # 随机亮度调整
    brightness_factor = random.uniform(*brightness_range)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    
    # 随机对比度调整
    contrast_factor = random.uniform(*contrast_range)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    
    # 随机饱和度调整
    saturation_factor = random.uniform(*saturation_range)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    
    # 随机色调调整（HSV空间）
    hue_factor = random.uniform(*hue_range)
    hsv = image.convert('HSV')
    H, S, V = hsv.split()
    H = H.point(lambda x: (x + hue_factor * 255) % 255)
    image = Image.merge('HSV', (H, S, V)).convert('RGB')
    
    return image

if __name__ == '__main__':
    process_dataset()
    print(f"数据增强完成！增强样本保存在 {output_dir} 目录中")