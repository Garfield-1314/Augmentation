import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_digits(font_dir, output_root,
                   digits_range=(0, 9),
                   total_samples=1000,
                   base_padding=2,
                   underline_config=None):
    """
    带文件名后缀的图片生成器
    
    参数说明：
    - font_dir: 字体文件目录路径
    - output_root: 输出根目录
    - digits_range: 生成数字范围 (起始, 结束)
    - total_samples: 每个子目录的样本数量
    - base_padding: 基础内边距（像素）
    - underline_config: 下划线配置字典（可选）
    """
    # 初始化字体库
    font_files = _load_valid_fonts(font_dir)
    if not font_files:
        raise ValueError(f"未找到有效字体文件：{font_dir}")

    # 创建输出目录
    _create_dirs(output_root, digits_range)

    # 样本分配计算
    samples_per_digit = _calculate_samples(digits_range, total_samples)

    # 下划线默认配置
    underline_config = underline_config or {
        'width': 2,
        'padding': 2
    }

    # 主生成循环（双版本）
    for version in ['normal', 'underlined']:
        for digit, sample_count in zip(range(*digits_range), samples_per_digit):
            for sample_idx in range(sample_count):
                _generate_digit_image(
                    digit=digit,
                    sample_idx=sample_idx,
                    font_files=font_files,
                    output_root=output_root,
                    base_padding=base_padding,
                    version=version,
                    underline_config=underline_config
                )

def _load_valid_fonts(font_dir):
    """加载验证有效字体文件"""
    valid_ext = ('.ttf', '.otf')
    return [
        os.path.join(font_dir, f)
        for f in os.listdir(font_dir)
        if f.lower().endswith(valid_ext) and _is_valid_font(os.path.join(font_dir, f))
    ]

def _is_valid_font(font_path):
    """验证字体有效性"""
    try:
        ImageFont.truetype(font_path, 10)
        return True
    except Exception:
        return False

def _create_dirs(output_root, digits_range):
    """创建新版目录结构"""
    # 创建版本主目录
    for version in ['normal', 'underlined']:
        version_dir = os.path.join(output_root, version)
        # 创建数字子目录
        for d in range(digits_range[0], digits_range[1]):
            os.makedirs(os.path.join(version_dir, str(d)), exist_ok=True)

def _calculate_samples(digits_range, total_samples):
    """计算每个数字的样本数量"""
    num_digits = digits_range[1] - digits_range[0]
    base_samples = total_samples // num_digits
    remainder = total_samples % num_digits
    return [base_samples + 1 if i < remainder else base_samples 
            for i in range(num_digits)]

def _generate_digit_image(digit, sample_idx, font_files, output_root, 
                         base_padding, version, underline_config):
    """生成单版本数字图像（含_U后缀）"""
    # 确定保存路径和文件名
    save_dir = os.path.join(output_root, version, str(digit))
    
    # 生成带版本标记的文件名
    filename = f"{digit}_{sample_idx:04d}_U.png" if version == 'underlined' \
               else f"{digit}_{sample_idx:04d}.png"
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # 随机选择字体
            font_path = random.choice(font_files)
            initial_size = 100  # 初始字体大小
            
            # 确保字体可用
            font, bbox = _find_proper_font(font_path, str(digit), initial_size)
            if not font:
                continue
                
            # 计算图片尺寸
            width = bbox[2] - bbox[0] + 2*base_padding
            height = bbox[3] - bbox[1] + 2*base_padding
            
            # 创建画布
            img = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 计算绘制位置
            x = -bbox[0] + base_padding
            y = -bbox[1] + base_padding
            draw.text((x, y), str(digit), font=font, fill=(0, 0, 0))
            
            # 条件性添加下划线
            if version == 'underlined':
                img = _add_smart_underline(img, bbox, base_padding, underline_config)
            
            # 保存文件
            img.save(os.path.join(save_dir, filename))
            return
            
        except Exception as e:
            if font_path in font_files:
                font_files.remove(font_path)
            if not font_files:
                raise RuntimeError("所有字体尝试失败")
    
    print(f"警告: 数字{digit} 样本{sample_idx} ({version}) 生成失败")

def _find_proper_font(font_path, char, initial_size):
    """寻找合适字体大小"""
    try:
        font = ImageFont.truetype(font_path, initial_size)
        bbox = font.getbbox(char)
        
        # 自动调整大小确保最小尺寸
        min_size = 20
        while (bbox[2] - bbox[0] < min_size or 
               bbox[3] - bbox[1] < min_size) and initial_size < 500:
            initial_size += 10
            font = ImageFont.truetype(font_path, initial_size)
            bbox = font.getbbox(char)
        
        return font, bbox
    except:
        return None, None

def _add_smart_underline(image, char_bbox, padding, config):
    """添加智能下划线"""
    width, height = image.size
    line_height = config['width'] + config['padding']
    new_height = height + line_height
    
    new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
    new_img.paste(image, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    y_position = height - padding + config['padding']
    draw.line(
        [(padding, y_position), 
         (width - padding, y_position)],
        fill=(0, 0, 0),
        width=config['width']
    )
    
    return new_img

if __name__ == '__main__':
    generate_digits(
        font_dir='./nums/fonts',
        output_root='./nums/dataset',
        digits_range=(0, 10),  # 生成0-9
        total_samples=100,
        base_padding=2,
        underline_config={
            'width': 4,
            'padding': 2
        }
    )