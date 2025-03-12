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
                 pics_root=r'dataset\YASUO_80',
                 output_root=r'dataset\merged_output'):
    """
    保留子目录结构的批量处理
    :param backgrounds_dir: 背景图目录
    :param pics_root: 小图根目录（包含子目录）
    :param output_root: 输出根目录
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
                # 计算相对路径
                rel_path = os.path.relpath(pic_path, pics_root)
                output_dir = os.path.join(output_root, os.path.dirname(rel_path))
                
                # 创建对应子目录
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                pic_name = os.path.splitext(os.path.basename(pic_path))[0]
                output_name = f"{bg_name}_{pic_name}_{timestamp}.jpg"
                output_path = os.path.join(output_dir, output_name)
                
                # 处理单个合成
                try:
                    small_img = Image.open(pic_path).convert('RGBA')
                    
                    # 随机旋转和定位
                    angle = random.uniform(0, 360)
                    rotated = small_img.rotate(angle, expand=True, fillcolor=(0,0,0,0))
                    rw, rh = rotated.size
                    
                    # 确保75%可见的定位逻辑
                    valid_pos = False
                    for _ in range(100):
                        x = random.randint(-int(rw*0.3), bg_w - int(rw*0.7))
                        y = random.randint(-int(rh*0.3), bg_h - int(rh*0.7))
                        visible_area = (min(x+rw, bg_w) - max(x,0)) * (min(y+rh, bg_h) - max(y,0))
                        if visible_area >= 0.75 * rw * rh:
                            valid_pos = True
                            break
                    
                    # 合成图像
                    composite = Image.new('RGBA', (bg_w, bg_h))
                    composite.paste(base_img, (0,0))
                    composite.alpha_composite(rotated, (x,y))
                    composite.convert('RGB').save(output_path)
                    
                    print(f"生成成功：{output_path}")
                    
                except Exception as e:
                    print(f"小图处理失败：{pic_path} | 错误：{str(e)}")
                
        except Exception as e:
            print(f"背景图处理失败：{bg_path} | 错误：{str(e)}")

if __name__ == '__main__':
    batch_overlay()
