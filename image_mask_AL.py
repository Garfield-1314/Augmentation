import os
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from albumentations import (
    Compose, Rotate, RGBShift, 
    RandomBrightnessContrast, MotionBlur,
    HueSaturationValue, ElasticTransform, 
    OpticalDistortion
)

def find_images(root_dir):
    """递归查找所有子目录中的图片文件（支持常见格式）"""
    img_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(img_extensions):
                yield os.path.join(dirpath, filename)

# 定义符合 2.0.5 版本的数据增强管道
augmentation_pipeline = Compose([
    # 弹性变形（添加边界反射模式）
    ElasticTransform(
        p=0.25,
        alpha=1.2,
        sigma=25,
    ),
    # 光学畸变（添加黑色填充）
    OpticalDistortion(
        p=0.25,
        distort_limit=0.25,
    ),
    # 随机旋转（镜像边界处理）
    Rotate(
        limit=20,
        p=0.6,
        border_mode=cv2.BORDER_REFLECT_101
    ),
    # RGB通道偏移
    RGBShift(
        r_shift_limit=15,
        g_shift_limit=15,
        b_shift_limit=15,
        p=0.3
    ),
    # 亮度对比度调整
    RandomBrightnessContrast(
        p=0.8,
        brightness_limit=(-0.3, 0.3),
        contrast_limit=(-0.15, 0.15)
    ),
    # 色相饱和度调整
    HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=15,
        p=0.4
    ),
    # 动态模糊（使用随机核大小）
    MotionBlur(
        p=0.3,
        blur_limit=(3, 9)
    )
], p=0.9)  # 整体应用概率

def batch_overlay(
    backgrounds_dir=r'dataset\background',
    pics_root=r'dataset\stage2',
    output_root=r'dataset\stage3',
    min_scale=0.3,
    max_scale=1.7,
    min_visible=0.6,
    num_augments=3
):
    """批量图像合成与增强
    
    参数说明：
    min_scale - 最小缩放比例（相对于原图）
    max_scale - 最大缩放比例
    min_visible - 最小可见区域比例
    num_augments - 每张图的增强次数
    """
    
    # 获取所有背景图和小图路径
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))
    print(f"发现 {len(bg_paths)} 张背景图，{len(pic_paths)} 张小图")

    # 处理每张背景图
    for bg_idx, bg_path in enumerate(bg_paths, 1):
        try:
            # 加载背景图
            base_img = Image.open(bg_path).convert('RGBA')
            bg_w, bg_h = base_img.size
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            print(f"\n处理背景图 [{bg_idx}/{len(bg_paths)}]: {bg_name}")

            # 处理每张小图
            for pic_idx, pic_path in enumerate(pic_paths, 1):
                try:
                    # 创建输出目录
                    rel_path = os.path.relpath(pic_path, pics_root)
                    output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                    os.makedirs(output_dir, exist_ok=True)

                    # 多次增强生成
                    for aug_idx in range(1, num_augments+1):
                        # 加载并预处理小图
                        small_img = Image.open(pic_path).convert('RGBA')
                        
                        # 随机缩放
                        scale_factor = random.uniform(min_scale, max_scale)
                        new_size = (
                            int(small_img.width * scale_factor),
                            int(small_img.height * scale_factor)
                        )
                        scaled_img = small_img.resize(new_size, Image.LANCZOS)

                        # 随机旋转（带透明填充）
                        rotation_angle = random.uniform(0, 360)
                        rotated_img = scaled_img.rotate(
                            rotation_angle,
                            expand=True,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0, 0)
                        )
                        rw, rh = rotated_img.size

                        # 智能定位算法（确保可见区域）
                        valid_position = False
                        for _ in range(100):  # 最多尝试100次定位
                            # 计算可用位置范围
                            x_min = max(-int(rw * 0.3), -rw + int(bg_w * 0.15))
                            x_max = min(bg_w - int(rw * 0.7), bg_w - int(rw * 0.15))
                            y_min = max(-int(rh * 0.3), -rh + int(bg_h * 0.15))
                            y_max = min(bg_h - int(rh * 0.7), bg_h - int(rh * 0.15))
                            
                            # 随机坐标
                            pos_x = random.randint(x_min, x_max)
                            pos_y = random.randint(y_min, y_max)
                            
                            # 计算可见区域
                            visible_width = min(pos_x + rw, bg_w) - max(pos_x, 0)
                            visible_height = min(pos_y + rh, bg_h) - max(pos_y, 0)
                            if visible_width > 0 and visible_height > 0:
                                visible_area = visible_width * visible_height
                                if visible_area >= min_visible * rw * rh:
                                    valid_position = True
                                    break

                        if not valid_position:
                            print(f"无法为 {os.path.basename(pic_path)} 找到有效位置")
                            continue

                        # 合成图像
                        composite = Image.new('RGBA', (bg_w, bg_h))
                        composite.paste(base_img, (0, 0))
                        composite.alpha_composite(rotated_img, (pos_x, pos_y))
                        rgb_composite = composite.convert('RGB')

                        # 转换为NumPy数组（保持RGB格式）
                        np_image = np.array(rgb_composite)

                        # 应用数据增强
                        augmented = augmentation_pipeline(image=np_image)
                        augmented_img = augmented['image']

                        # 转换为BGR格式保存
                        bgr_image = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

                        # 生成唯一文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
                        pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                        output_filename = (
                            f"{bg_name}_{pic_name}_{timestamp}_aug{aug_idx}.jpg"
                        )
                        output_path = os.path.join(output_dir, output_filename)

                        # 保存图像（优化JPEG质量）
                        cv2.imwrite(
                            output_path, 
                            bgr_image, 
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                        )
                        print(f"已生成：{output_path}")

                except Exception as e:
                    print(f"处理失败：{pic_path} | 错误：{str(e)}")

        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    # 示例配置（可根据需要调整）
    batch_overlay(
        backgrounds_dir=r'dataset\background',
        pics_root=r'images',
        output_root=r'Au',
        min_scale=0.8,
        max_scale=1.2,
        min_visible=0.5,
        num_augments=2
    )