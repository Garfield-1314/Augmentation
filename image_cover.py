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
                 min_scale=0.3,  # 新增缩放参数
                 max_scale=1.7):
    """
    支持缩放、旋转和位置调整的批量处理
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
                    # 加载并处理小图
                    small_img = Image.open(pic_path).convert('RGBA')
                    
                    # 随机缩放（保持宽高比）
                    scale = random.uniform(min_scale, max_scale)
                    new_size = (
                        int(small_img.width * scale),
                        int(small_img.height * scale)
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
                    
                    # 智能定位（至少75%可见）
                    valid_pos = False
                    for _ in range(100):
                        # 动态计算位置范围
                        x_min = max(-int(rw * 0.3), -rw + int(bg_w * 0.25))
                        x_max = min(bg_w - int(rw * 0.7), bg_w - int(rw * 0.25))
                        y_min = max(-int(rh * 0.3), -rh + int(bg_h * 0.25))
                        y_max = min(bg_h - int(rh * 0.7), bg_h - int(rh * 0.25))
                        
                        x = random.randint(x_min, x_max)
                        y = random.randint(y_min, y_max)
                        
                        # 计算可见区域
                        visible_w = min(x+rw, bg_w) - max(x, 0)
                        visible_h = min(y+rh, bg_h) - max(y, 0)
                        if visible_w > 0 and visible_h > 0:
                            visible_area = visible_w * visible_h
                            if visible_area >= 0.75 * rw * rh:
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
    for i in range(10):
        batch_overlay(
            backgrounds_dir=r'dataset\background', 
            pics_root=r'dataset\80',
            output_root=r'dataset\COVER',
            min_scale=0.8,  # 可调节参数
            max_scale=1.2   # 缩放范围 50%-150%
        )
