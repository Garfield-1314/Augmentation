import os
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import albumentations as A

def find_images(root_dir):
    """递归查找所有子目录中的图片文件"""
    img_ext = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(img_ext):
                yield os.path.join(dirpath, f)

# 定义数据增强管道
augmentation_pipeline = A.Compose([
    A.ElasticTransform(p=0.45, alpha=1.2, sigma=50),
    A.OpticalDistortion(p=0.45, distort_limit=0.25),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25)),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.6),
    A.MotionBlur(p=0.3, blur_limit=(3))
])

def batch_overlay(
    backgrounds_dir=r'dataset\background',
    pics_root=r'dataset\stage2',
    output_root=r'dataset\stage3',
    min_scale=0.3,
    max_scale=1.7,
    num_augments=3
):
    # 获取所有背景和小图路径
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))

    # 处理每个组合
    for bg_path in bg_paths:
        try:
            base_img = Image.open(bg_path).convert('RGBA')
            bg_w, bg_h = base_img.size
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            
            for pic_path in pic_paths:
                rel_path = os.path.relpath(pic_path, pics_root)
                output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                os.makedirs(output_dir, exist_ok=True)

                try:
                    for aug_idx in range(num_augments):
                        small_img = Image.open(pic_path).convert('RGBA')
                        
                        # 动态计算最大缩放比例
                        max_scale_x = bg_w / small_img.width
                        max_scale_y = bg_h / small_img.height
                        actual_max_scale = min(max_scale, max_scale_x, max_scale_y)
                        scale = random.uniform(min_scale, actual_max_scale)
                        
                        new_size = (int(small_img.width * scale), int(small_img.height * scale))
                        scaled_img = small_img.resize(new_size, Image.LANCZOS)
                        
                        # 随机旋转
                        angle = random.uniform(0, 360)
                        rotated_img = scaled_img.rotate(
                            angle,
                            expand=True,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0, 0)
                        )
                        rw, rh = rotated_img.size
                        
                        # 计算有效位置范围
                        x_min, x_max = 0, bg_w - rw
                        y_min, y_max = 0, bg_h - rh
                        valid_pos = False

                        if x_max >= x_min and y_max >= y_min:
                            x = random.randint(x_min, x_max)
                            y = random.randint(y_min, y_max)
                            valid_pos = True

                        if not valid_pos:
                            continue

                        # 合成图像
                        composite = Image.new('RGBA', (bg_w, bg_h))
                        composite.paste(base_img, (0,0))
                        composite.alpha_composite(rotated_img, (x, y))
                        rgb_composite = composite.convert('RGB')
                        
                        # 转换为OpenCV格式
                        cv_image = cv2.cvtColor(np.array(rgb_composite), cv2.COLOR_RGB2BGR)
                        
                        # 应用数据增强
                        augmented = augmentation_pipeline(image=cv_image)
                        augmented_img = augmented['image']
                        
                        # 生成唯一文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                        output_name = f"{bg_name}_{pic_name}_{timestamp}_aug{aug_idx}.jpg"
                        output_path = os.path.join(output_dir, output_name)
                        
                        cv2.imwrite(output_path, augmented_img)
                        print(f"生成成功：{output_path}")

                except Exception as e:
                    print(f"处理失败：{pic_path} | 错误：{str(e)}")
                
        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    batch_overlay(
        backgrounds_dir='./background',
        pics_root='./nums/dataset2',
        output_root='./dataset/train',
        min_scale=0.6,
        max_scale=0.9,
        num_augments=40
    )