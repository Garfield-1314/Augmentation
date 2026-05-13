import os
import cv2
from albumentations import (
    Compose, HorizontalFlip, Rotate,
    RGBShift, RandomBrightnessContrast,MotionBlur,VerticalFlip,HueSaturationValue,ElasticTransform,OpticalDistortion
)
from tqdm import tqdm

def get_augmentation_pipeline():
    """定义并返回数据增强管道"""
    return Compose([
        # HorizontalFlip(p=0.25),
        # VerticalFlip(p=0.25),
        ElasticTransform(p=0.35, alpha=1, sigma=20),
        # OpticalDistortion(p=0.35,distort_limit=0.2, shift_limit=0.2),
        # Rotate(limit=45, p=0.5),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.25),
        RandomBrightnessContrast(p=0.25, brightness_limit=(-0.1, 0.15), contrast_limit=(-0.10, 0.10)),
        HueSaturationValue(hue_shift_limit=(-5, 5),
                           sat_shift_limit=(-5, 5),
                           val_shift_limit=(-5, 5),
                           p=0.25),
        # MotionBlur(p=0.25,blur_limit = 3),
    ])

def augment_images(input_dir, output_dir, num_augments=9):
    """
    遍历目录并对图片进行数据增强
    Args:
        input_dir: 输入图片根目录
        output_dir: 输出图片根目录
        num_augments: 每张图片生成的增强版本数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Applying advanced augmentations to {output_dir}...")
    
    # 获取增强管道
    augmentation_pipeline = get_augmentation_pipeline()

    # 支持的图片格式
    extensions = ['.jpg', '.jpeg', '.png']

    # 使用os.walk递归遍历所有子目录
    for root, dirs, files in os.walk(input_dir):
        # 计算相对路径以便重建输出目录结构
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        
        # 创建当前层级的输出目录
        os.makedirs(current_output_dir, exist_ok=True)
        
        # 处理当前目录下的所有文件
        for filename in tqdm(files, desc=f"Processing {relative_path}"):
            # 过滤非图片文件
            if not any(filename.lower().endswith(ext) for ext in extensions):
                continue
            
            # 读取图片
            img_path = os.path.join(root, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # 保存原始图片
            original_output = os.path.join(current_output_dir, f"original_{filename}")
            cv2.imwrite(original_output, image)
            
            # 生成多个增强版本
            for i in range(num_augments):
                augmented = augmentation_pipeline(image=image)
                augmented_img = augmented['image']
                
                # 构建增强后的文件名
                base_name, ext = os.path.splitext(filename)
                aug_filename = f"{base_name}_aug{i+1}{ext}"
                output_path = os.path.join(current_output_dir, aug_filename)
                
                # 保存增强后的图片
                cv2.imwrite(output_path, augmented_img)

if __name__ == "__main__":
    # 默认配置参数
    CONFIG = {
        "input_dir": "../Datasets/9_dataset_3",
        "output_dir": "../Datasets/99_dataset",
        "num_augments": 9
    }
    
    augment_images(**CONFIG)