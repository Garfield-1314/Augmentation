import os
import random
from PIL import Image
from datetime import datetime

def find_images(root_dir):
    """递归查找所有子目录中的图片文件"""
    img_ext = ('.png', '.jpg', '.jpeg', '.webp')
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(img_ext):
                yield os.path.join(dirpath, f)

def batch_overlay(backgrounds_dir=r'dataset\background', 
                 pics_root=r'dataset\stage2',
                 output_root=r'dataset\stage3',
                 min_scale=0.3,
                 max_scale=1.7,
                 min_visible=0.75,
                 center_mode=False):
    """
    支持缩放、旋转和位置调整的批量处理
    
    参数:
    min_visible - 可见区域最小比例 (0.0~1.0)
    center_mode - True: 小图居中无旋转变换; False: 随机变换模式
    """
    
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
                # 计算输出路径
                rel_path = os.path.relpath(pic_path, pics_root)
                output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                output_name = f"{bg_name}_{pic_name}_{timestamp}.jpg"
                output_path = os.path.join(output_dir, output_name)
                
                try:
                    # 加载小图
                    small_img = Image.open(pic_path).convert('RGBA')
                    
                    if center_mode:
                        # 居中模式 - 支持随机缩放但不旋转
                        scale = random.uniform(min_scale, max_scale)
                        new_size = (
                            max(1, int(small_img.width * scale)),
                            max(1, int(small_img.height * scale))
                        )
                        scaled_img = small_img.resize(new_size, Image.LANCZOS)
                        
                        # 居中放置
                        x = (bg_w - scaled_img.width) // 2
                        y = (bg_h - scaled_img.height) // 2
                        rotated_img = scaled_img
                        
                    else:
                        # 随机变换模式
                        # 随机缩放（保持宽高比）
                        scale = random.uniform(min_scale, max_scale)
                        new_size = (
                            max(1, int(small_img.width * scale)),
                            max(1, int(small_img.height * scale))
                        )
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
                        
                        # 智能定位（根据min_visible参数控制可见比例）
                        valid_pos = False
                        for _ in range(100):
                            # 动态计算位置范围
                            x_min = max(-int(rw * 0.3), -rw + int(bg_w * 0.25))
                            x_max = min(bg_w - int(rw * 0.7), bg_w - int(rw * 0.25))
                            y_min = max(-int(rh * 0.3), -rh + int(bg_h * 0.25))
                            y_max = min(bg_h - int(rh * 0.7), bg_h - int(rh * 0.25))
                            
                            # 确保位置范围有效
                            if x_min > x_max:
                                x_min, x_max = -rw, bg_w
                            if y_min > y_max:
                                y_min, y_max = -rh, bg_h
                            
                            x = random.randint(x_min, x_max)
                            y = random.randint(y_min, y_max)
                            
                            # 计算可见区域
                            visible_w = min(x+rw, bg_w) - max(x, 0)
                            visible_h = min(y+rh, bg_h) - max(y, 0)
                            if visible_w > 0 and visible_h > 0:
                                visible_area = visible_w * visible_h
                                # 使用参数控制可见比例
                                if visible_area >= min_visible * rw * rh:
                                    valid_pos = True
                                    break
                    
                    # 合成图像
                    composite = Image.new('RGBA', (bg_w, bg_h))
                    composite.paste(base_img, (0,0))
                    composite.alpha_composite(rotated_img, (x, y))
                    composite.convert('RGB').save(output_path)
                    
                    print(f"生成成功：{output_path}")
                    
                except Exception as e:
                    print(f"小图处理失败：{pic_path} | 错误：{str(e)}")
                
        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    # 示例调用 - 使用居中模式并调整缩放比例
    batch_overlay(
        backgrounds_dir='../Datasets/background_w_135', 
        pics_root='../Datasets/9_dataset_3',
        output_root='../Datasets/9_dataset_3_135',
        min_scale=0.8,  # 居中模式下的最小缩放比例
        max_scale=1.2,  # 居中模式下的最大缩放比例
        min_visible=1.0,
        center_mode=True  # 启用居中模式
    )