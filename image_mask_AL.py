import os
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import albumentations as A
from tqdm import tqdm

def find_images(root_dir):
    """递归查找所有子目录中的图片文件"""
    img_ext = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(img_ext):
                yield os.path.join(dirpath, f)

# 定义小图增强管道（仅颜色变换）
small_aug_pipeline = A.Compose([
    A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.10, 0.10), contrast_limit=(-0.1, 0.1)),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), 
                         sat_shift_limit=(-15, 15), 
                         val_shift_limit=(-10, 10), 
                         p=0.7)
])

# 定义全局增强管道（完整变换）
global_aug_pipeline = A.Compose([
    A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p=0.5),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), 
                         sat_shift_limit=(-15, 15), 
                         val_shift_limit=(-10, 10), 
                         p=0.7),
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.10, 0.10), contrast_limit=(-0.1, 0.1)),
    A.ElasticTransform(p=0.1, alpha=1.2, sigma=50),
    A.Rotate(limit=(-5,5), border_mode=cv2.BORDER_WRAP, p=0.5),
    A.MotionBlur(p=0.1, blur_limit=(3)),
    A.ISONoise(p=0.2),
    # A.CoarseDropout(
    #     num_holes_range=(1,2),          # 最大遮挡块数量
    #     hole_height_range=(0.05,0.1),  
    #     hole_width_range=(0.05,0.1),
    #     fill_value = 0,
    #     p=0.1                # 应用概率
    # )
])

def apply_small_aug(img_cv):
    """对小图应用增强并返回增强后的OpenCV图像"""
    # 应用增强管道
    augmented = small_aug_pipeline(image=img_cv)
    return augmented['image']

def batch_overlay(
    backgrounds_dir=r'dataset\background',
    pics_root=r'dataset\stage2',
    output_root=r'dataset\stage3',
    min_scale=0.3,
    max_scale=1.7,
    min_visible=0.75,  # 控制小图在ROI内的可见面积比例
    num_augments=3,
    roi_list=None
):
    # 获取所有背景和小图路径
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))
    
    # 构建背景文件路径到ROI的映射
    roi_dict = {}
    if roi_list:
        for bg_path, roi in roi_list:
            bg_full_path = os.path.join(backgrounds_dir, bg_path) if not os.path.isabs(bg_path) else bg_path
            roi_dict[bg_full_path] = roi
    
    print(f"找到 {len(bg_paths)} 张背景图片")
    print(f"找到 {len(pic_paths)} 张小图")
    total_tasks = len(bg_paths) * len(pic_paths) * num_augments
    print(f"总任务量: {total_tasks} 张合成图")
    
    # 初始化进度条
    pbar = tqdm(total=total_tasks, desc="合成进度", unit="image", dynamic_ncols=True)

    # 处理每个组合
    for bg_path in bg_paths:
        try:
            base_img = Image.open(bg_path).convert('RGBA')
            bg_w, bg_h = base_img.size
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            
            # 获取当前背景的ROI，如果没有则使用整个背景
            roi = roi_dict.get(bg_path, None)
            if roi is None:
                roi = (0, 0, bg_w, bg_h)  # 默认使用全图作为ROI
            roi_x, roi_y, roi_w, roi_h = roi
            
            # 验证ROI是否在图像边界内
            roi_x = max(0, min(roi_x, bg_w - 1))
            roi_y = max(0, min(roi_y, bg_h - 1))
            roi_w = max(1, min(roi_w, bg_w - roi_x))
            roi_h = max(1, min(roi_h, bg_h - roi_y))
            
            for pic_path in pic_paths:
                try:
                    # 加载小图并转换为OpenCV格式 (RGB)
                    small_img_pil = Image.open(pic_path).convert('RGBA')
                    small_img_cv = cv2.cvtColor(np.array(small_img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                    
                    # 应用小图增强
                    small_img_cv = apply_small_aug(small_img_cv)
                    
                    # 转换回PIL格式 (RGBA)
                    small_img_rgb = cv2.cvtColor(small_img_cv, cv2.COLOR_BGR2RGB)
                    small_img_pil = Image.fromarray(small_img_rgb).convert('RGBA')
                    
                    for aug_idx in range(num_augments):
                        # 随机缩放
                        scale = random.uniform(min_scale, max_scale)
                        new_size = (int(small_img_pil.width * scale), int(small_img_pil.height * scale))
                        scaled_img = small_img_pil.resize(new_size, Image.LANCZOS)

                        # 随机旋转
                        angle = random.choice([0, 90, 180, 270])
                        angle1 = random.uniform(-10, 10)
                        angle = angle + angle1
                        rotated_img = scaled_img.rotate(
                            angle,
                            expand=True,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0, 0)
                        )
                        rw, rh = rotated_img.size
                        
                        # 智能定位 - 专门在ROI区域内放置小图
                        valid_pos = False
                        for _ in range(100):  # 最多尝试100次寻找有效位置
                            # 计算小图左上角坐标范围（确保中心点在ROI内）
                            center_x_min = roi_x + (min_visible * rw) / 2
                            center_x_max = roi_x + roi_w - (min_visible * rw) / 2
                            center_y_min = roi_y + (min_visible * rh) / 2
                            center_y_max = roi_y + roi_h - (min_visible * rh) / 2
                            
                            # 确保范围有效
                            if center_x_min > center_x_max or center_y_min > center_y_max:
                                continue
                            
                            # 随机选择中心点位置
                            center_x = random.uniform(center_x_min, center_x_max)
                            center_y = random.uniform(center_y_min, center_y_max)
                            
                            # 从中心点计算左上角位置
                            x = int(center_x - rw / 2)
                            y = int(center_y - rh / 2)
                            
                            # 计算小图与ROI的交集区域
                            intersect_x1 = max(x, roi_x)
                            intersect_y1 = max(y, roi_y)
                            intersect_x2 = min(x + rw, roi_x + roi_w)
                            intersect_y2 = min(y + rh, roi_y + roi_h)
                            
                            # 计算交集面积和小图总面积
                            intersection_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)
                            small_img_area = rw * rh
                            
                            # 检查可见面积占比是否符合要求
                            if intersection_area / small_img_area >= min_visible:
                                valid_pos = True
                                break

                        if not valid_pos:
                            # 如果找不到有效位置，跳过当前增强
                            print(f"无法为 {pic_path} 找到满足 min_visible={min_visible} 的位置")
                            continue

                        # 计算输出路径
                        rel_path = os.path.relpath(pic_path, pics_root)
                        output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 合成基础图像
                        composite = Image.new('RGBA', (bg_w, bg_h))
                        composite.paste(base_img, (0,0))
                        composite.alpha_composite(rotated_img, (x, y))
                        rgb_composite = composite.convert('RGB')
                        
                        # 转换为OpenCV格式
                        cv_image = cv2.cvtColor(np.array(rgb_composite), cv2.COLOR_RGB2BGR)
                        
                        # 应用全局数据增强
                        augmented = global_aug_pipeline(image=cv_image)
                        augmented_img = augmented['image']
                        
                        # 生成唯一文件名
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                        pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                        output_name = f"{bg_name}_{pic_name}_{timestamp}_aug{aug_idx}.jpg"
                        output_path = os.path.join(output_dir, output_name)
                        
                        # 保存增强后的图像
                        cv2.imwrite(output_path, augmented_img)
                        
                        # 更新进度条
                        pbar.set_postfix_str(f"处理: {os.path.basename(output_path)}")
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"处理失败：{pic_path} | 错误：{str(e)}")
                
        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")
    
    pbar.close()
    print("所有图像合成完成！")

if __name__ == '__main__':
    # 示例用法1：没有指定ROI，默认使用整个背景
    # batch_overlay(
    #     backgrounds_dir='./background',
    #     pics_root='./SC20_120',
    #     output_root='./data_obj_1',
    #     min_scale=0.9,
    #     max_scale=1.1,
    #     min_visible=0.85,  # 85%的小图必须位于ROI区域内
    #     num_augments=9
    # )
    
    # 示例用法2：指定ROI - 格式为[(背景路径, (x, y, w, h)), ...]
    custom_roi = [
        ('./background',(100,100,24,24)),    # 为bg1.jpg指定ROI
    ]
    
    batch_overlay(
        backgrounds_dir='../Datasets/background',
        pics_root='../Datasets/SC20_135',
        output_root='../Datasets/SC20_135_roi/val',
        min_scale=0.9,
        max_scale=1.1,
        min_visible=1.0,  # 80%的小图必须位于指定ROI区域内
        num_augments=6,
        roi_list=custom_roi
    )