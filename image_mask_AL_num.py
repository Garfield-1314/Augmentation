import os
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import albumentations as A

def find_images(root_dir):
    """递归查找所有子目录中的图片文件（支持常见格式）"""
    img_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(img_extensions):
                yield os.path.join(dirpath, filename)

# 定义数据增强管道
augmentation_pipeline = A.Compose([
    A.ElasticTransform(p=0.45, alpha=1.2, sigma=50),
    A.OpticalDistortion(p=0.45, distort_limit=0.25),
    A.Rotate(limit=10, p=0.6, border_mode=cv2.INTER_NEAREST),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.5, 0.5), contrast_limit=(-0.25, 0.25)),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.6),
    A.MotionBlur(p=0.3, blur_limit=(3, 7))
], p=0.9)

def calculate_max_scale(orig_size, bg_size, min_visible):
    """智能计算最大缩放比例"""
    orig_w, orig_h = orig_size
    bg_w, bg_h = bg_size
    max_area_scale = np.sqrt((bg_w * bg_h * min_visible) / (orig_w * orig_h))
    max_width_scale = bg_w / orig_w
    max_height_scale = bg_h / orig_h
    return min(max_area_scale, max_width_scale, max_height_scale)

def try_positioning(rotated_img, bg_size, min_visible, attempts=100):
    """尝试定位的可复用函数"""
    rw, rh = rotated_img.size
    bg_w, bg_h = bg_size
    
    for _ in range(attempts):
        x = random.randint(-rw//2, bg_w - rw//2)
        y = random.randint(-rh//2, bg_h - rh//2)
        
        vis_left = max(x, 0)
        vis_right = min(x + rw, bg_w)
        vis_width = vis_right - vis_left
        
        vis_top = max(y, 0)
        vis_bottom = min(y + rh, bg_h)
        vis_height = vis_bottom - vis_top
        
        if vis_width <= 0 or vis_height <= 0:
            continue
        
        visible_ratio = (vis_width * vis_height) / (rw * rh)
        if visible_ratio >= min_visible:
            return (x, y), True
    return None, False

def batch_overlay(
    backgrounds_dir='./background',
    pics_root='./nums/dataset',
    output_root='./dataset/train',
    min_scale=0.5,
    max_scale=1.5,
    min_visible=0.7,
    num_augments=100
):
    """带强制定位功能的图像合成"""
    
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))
    print(f"发现 {len(bg_paths)} 张背景图，{len(pic_paths)} 张小图")

    for bg_idx, bg_path in enumerate(bg_paths, 1):
        try:
            base_img = Image.open(bg_path).convert('RGBA')
            bg_w, bg_h = base_img.size
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            print(f"\n处理背景图 [{bg_idx}/{len(bg_paths)}]: {bg_name}")

            for pic_path in pic_paths:
                try:
                    rel_path = os.path.relpath(pic_path, pics_root)
                    output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                    os.makedirs(output_dir, exist_ok=True)

                    small_img = Image.open(pic_path).convert('RGBA')
                    orig_w, orig_h = small_img.size

                    # 计算有效缩放范围
                    max_possible = calculate_max_scale(
                        (orig_w, orig_h), 
                        (bg_w, bg_h),
                        min_visible
                    )
                    effective_max = min(max_scale, max_possible)
                    effective_min = max(min_scale, 0.01)

                    if effective_max < effective_min:
                        print(f"⚠️ 跳过 {os.path.basename(pic_path)}：无效缩放范围")
                        continue

                    for aug_idx in range(num_augments):
                        # 初始随机参数
                        scale = random.uniform(effective_min, effective_max)
                        angle = random.uniform(-15, 15)
                        success = False

                        # 递归调整函数
                        def attempt_adjustment(current_scale, attempt=0):
                            nonlocal success
                            if attempt > 5 or current_scale < effective_min:
                                return None

                            # 缩放和旋转
                            new_w = int(orig_w * current_scale)
                            new_h = int(orig_h * current_scale)
                            scaled_img = small_img.resize((new_w, new_h), Image.LANCZOS)
                            rotated_img = scaled_img.rotate(
                                angle,
                                expand=True,
                                resample=Image.BICUBIC,
                                fillcolor=(0, 0, 0, 0)
                            )
                            
                            # 尝试定位
                            position, found = try_positioning(rotated_img, (bg_w, bg_h), min_visible)
                            
                            if not found:
                                # 递归调整缩放
                                return attempt_adjustment(current_scale * 0.9, attempt + 1)
                            else:
                                success = True
                                return (rotated_img, position, current_scale)

                        # 首次尝试
                        result = attempt_adjustment(scale)
                        
                        if not success:
                            print(f"⏩ 无法定位 {os.path.basename(pic_path)}，跳过")
                            continue

                        # 解包结果
                        rotated_img, (x, y), final_scale = result
                        # print(f"✅ 使用缩放比例 {final_scale:.2f} 成功定位")

                        # 合成图像
                        composite = Image.new('RGBA', (bg_w, bg_h))
                        composite.paste(base_img, (0, 0))
                        composite.alpha_composite(rotated_img, (x, y))
                        
                        # 数据增强
                        np_img = np.array(composite.convert('RGB'))
                        augmented = augmentation_pipeline(image=np_img)
                        bgr_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                        # 保存结果
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
                        pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                        output_filename = f"{bg_name}_{pic_name}_{timestamp}_aug{aug_idx+1}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        cv2.imwrite(output_path, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        print(f"✅ 已生成：{output_path}")

                except Exception as e:
                    print(f"❌ 处理失败：{pic_path} | 错误：{str(e)}")

        except Exception as e:
            print(f"❌ 背景处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    # 示例配置（可根据需要调整）
    batch_overlay(
        backgrounds_dir='./background',
        pics_root='96',
        output_root='image',
        min_scale=1.2,
        max_scale=1.4,
        min_visible=0.8,
        num_augments=10
    )