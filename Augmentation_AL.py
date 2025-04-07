import os
import cv2
from albumentations import (
    Compose, HorizontalFlip, Rotate,
    RGBShift, RandomBrightnessContrast,MotionBlur,VerticalFlip,HueSaturationValue,ElasticTransform,OpticalDistortion
)
from tqdm import tqdm

# 配置参数
input_dir = r"dataset\96"        # 输入图片根目录（包含子文件夹）
output_dir = r"dataset\al"   # 输出图片根目录
num_augments = 4 - 1                   # 每张图片生成多少个增强版本

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 定义数据增强管道
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.25),
    VerticalFlip(p=0.25),
    ElasticTransform(p=0.25,alpha=1, sigma=20),
    OpticalDistortion(p=0.25,distort_limit=0.1, shift_limit=0.1),
    Rotate(limit=45, p=0.5),
    RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.25),
    RandomBrightnessContrast(p=1,brightness_limit=(-0.25,0.25),contrast_limit=(-0.10,0.10)),
    HueSaturationValue( hue_shift_limit = (-10, 10),
                        sat_shift_limit = (-10, 10),
                        val_shift_limit = (-10, 10),
                        p=0.25),
    MotionBlur(p=0.25,blur_limit = 3),
])

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
        
        # 保存原始图片（可选）
        original_output = os.path.join(current_output_dir, f"original_{filename}")
        cv2.imwrite(original_output, image)
        
        # 生成多个增强版本
        for i in range(num_augments):
            augmented = augmentation_pipeline(image=image)
            augmented_img = augmented['image']
            
            # 构建增强后的文件名
            aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(current_output_dir, aug_filename)
            
            # 保存增强后的图片
            cv2.imwrite(output_path, augmented_img)