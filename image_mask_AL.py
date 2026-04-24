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

# 定义小图增强管道（仅颜色变换，针对 RGB）
small_aug_pipeline = A.Compose([
    A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.10, 0.10), contrast_limit=(-0.1, 0.1)),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), 
                         sat_shift_limit=(-15, 15), 
                         val_shift_limit=(-10, 10), 
                         p=0.7)
])

# 定义专门处理带 Alpha 通道的几何变换管道
small_geom_pipeline = A.Compose([
    # 使用 Affine 的 shear (错切) 来模拟明显的侧视效果
    # rotate=(-10, 10) 模拟微小旋转
    # shear={'x': (-20, 20)} 产生水平方向的拉伸位移，模拟侧方观察视角
    A.Affine(
        shear={'x': (-10, 10), 'y': (-10, 10)}, 
        rotate=(-10, 10),
        fit_output=True, 
        p=0.8,
        cval=0
    )
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
    # A.Rotate(limit=(-5,5), border_mode=cv2.BORDER_WRAP, p=0.5),
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

def apply_small_aug(img_pil_rgba):
    """对小图应用增强（分层处理颜色与几何变换）"""
    # 1. 颜色增强阶段：仅对 RGB 部分操作
    img_cv_rgba = np.array(img_pil_rgba)
    img_cv_rgb = cv2.cvtColor(img_cv_rgba, cv2.COLOR_RGBA2RGB)
    
    # 应用颜色增强
    augmented_color = small_aug_pipeline(image=img_cv_rgb)
    img_cv_rgb_aug = augmented_color['image']
    
    # 合并回 RGBA
    aug_rgba_data = cv2.cvtColor(img_cv_rgb_aug, cv2.COLOR_RGB2RGBA)
    aug_rgba_data[:, :, 3] = img_cv_rgba[:, :, 3]  # 保留原始 Alpha
    
    # 2. 几何增强阶段：对 BGRA 操作（支持透视后的透明填充）
    img_cv_bgra = cv2.cvtColor(aug_rgba_data, cv2.COLOR_RGBA2BGRA)
    augmented_geom = small_geom_pipeline(image=img_cv_bgra)
    img_cv_bgra_aug = augmented_geom['image']
    
    # 转换回 PIL RGBA
    img_cv_rgba_final = cv2.cvtColor(img_cv_bgra_aug, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(img_cv_rgba_final)

def batch_overlay(
    backgrounds_dir=r'dataset\background',
    pics_root=r'dataset\stage2',
    output_root=r'dataset\stage3',
    min_scale=0.3,
    max_scale=1.7,
    min_visible=0.75,  # 控制小图在ROI内的可见面积比例
    num_augments=3
):
    # 获取所有背景和小图路径
    bg_paths = list(find_images(backgrounds_dir))
    pic_paths = list(find_images(pics_root))
    
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
            
            # 使用整个背景作为放置区域
            roi_x, roi_y, roi_w, roi_h = (0, 0, bg_w, bg_h)
            
            for pic_path in pic_paths:
                try:
                    # 加载小图
                    small_img_pil = Image.open(pic_path).convert('RGBA')
                    
                    for aug_idx in range(num_augments):
                        # 应用小图增强（包含颜色和透视变换）
                        # 每次循环重新应用以保证随机性
                        current_small_img = apply_small_aug(small_img_pil)
                        
                        # 随机缩放
                        scale = random.uniform(min_scale, max_scale)
                        new_size = (int(current_small_img.width * scale), int(current_small_img.height * scale))
                        scaled_img = current_small_img.resize(new_size, Image.LANCZOS)

                        # 随机旋转
                        angle = random.choice([0, 90, 180, 270])
                        angle1 = random.uniform(-10, 10)
                        angle = 0
                        angle1 = 0
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
                        
                        # 重构可见度逻辑：
                        # min_visible 控制图片在 ROI (当前为全图) 内的最小边长比例
                        # 例如 0.9 表示图片在宽和高方向上至少有 90% 的长度落在背景内
                        
                        # 计算图片在左/上侧允许超出的最大长度
                        max_offset_x = rw * (1 - min_visible)
                        max_offset_y = rh * (1 - min_visible)
                        
                        # 计算左上角坐标 x, y 的允许范围：
                        # 最小值：图片左边缘在背景左边缘左侧 max_offset_x 处
                        # 最大值：图片右边缘在背景右边缘右侧 max_offset_x 处 (即 x = bg_w - rw + max_offset_x)
                        x_min = roi_x - max_offset_x
                        x_max = roi_x + roi_w - rw + max_offset_x
                        
                        y_min = roi_y - max_offset_y
                        y_max = roi_y + roi_h - rh + max_offset_y
                        
                        # 如果图片太大且 min_visible 要求很高导致逻辑冲突，则居中处理
                        if x_min > x_max:
                            x_min = x_max = roi_x + (roi_w - rw) / 2
                        if y_min > y_max:
                            y_min = y_max = roi_y + (roi_h - rh) / 2

                        # 随机选择左上角起始位置
                        x = int(random.uniform(x_min, x_max))
                        y = int(random.uniform(y_min, y_max))
                        valid_pos = True

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
    # custom_roi = [
    #     ('./background',(100,100,24,24)),    # 为bg1.jpg指定ROI
    # ]
    
    batch_overlay(
        backgrounds_dir='../Datasets/background',
        pics_root='../Datasets/smartcar26_160_pixelated',
        output_root='../Datasets/smartcar26_160_pixelated_masked_AL',
        min_scale=0.6,
        max_scale=1.1,
        min_visible=0.9,  # 100%的小图必须位于指定ROI区域内
        num_augments=10,
    )