import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_digits(font_dir, output_root,
                   digits_range=(0, 9),
                   total_samples=1000,
                   base_padding=2,
                   underline_config=None,
                   morph_config=None):
    """
    修改版数字生成器：
    - 仅生成 normal 和 thin 版本
    - 普通/细体版本存于 base 目录
    - 带下划线版本存于 underlined 目录
    - 文件名后缀规则：
        base目录：_N（普通）、_T（细体）
        underlined目录：_UN（普通下划线）、_UT（细体下划线）
    """
    # 初始化字体库
    font_files = _load_valid_fonts(font_dir)
    if not font_files:
        raise ValueError(f"未找到有效字体文件：{font_dir}")

    # 创建输出目录
    _create_dirs(output_root, digits_range)

    # 样本分配计算
    samples_per_digit = _calculate_samples(digits_range, total_samples)

    # 配置默认值
    underline_config = underline_config or {'width': 2, 'padding': 2}
    morph_config = morph_config or {
        'thin': {'operation': 'erode', 'kernel_size': 3, 'iterations': 1}
    }

    # 主生成循环（四版本组合）
    version_map = {
        'base_N': {'morph': None, 'underline': False},
        'base_T': {'morph': 'thin', 'underline': False},
        'underlined_UN': {'morph': None, 'underline': True},
        'underlined_UT': {'morph': 'thin', 'underline': True}
    }

    for version, config in version_map.items():
        for digit, sample_count in zip(range(*digits_range), samples_per_digit):
            for sample_idx in range(sample_count):
                _generate_digit_image(
                    digit=digit,
                    sample_idx=sample_idx,
                    font_files=font_files,
                    output_root=output_root,
                    base_padding=base_padding,
                    version_config=(version, config),
                    underline_config=underline_config,
                    morph_config=morph_config
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
    for category in ['base', 'underlined']:
        category_dir = os.path.join(output_root, category)
        for d in range(digits_range[0], digits_range[1]):
            os.makedirs(os.path.join(category_dir, str(d)), exist_ok=True)

def _calculate_samples(digits_range, total_samples):
    """计算每个数字的样本数量"""
    num_digits = digits_range[1] - digits_range[0]
    base_samples = total_samples // num_digits
    remainder = total_samples % num_digits
    return [base_samples + 1 if i < remainder else base_samples 
            for i in range(num_digits)]

def _generate_digit_image(digit, sample_idx, font_files, output_root, 
                         base_padding, version_config, underline_config, morph_config):
    """生成单版本数字图像"""
    version, config = version_config
    category = version.split('_')[0]
    suffix = version.split('_')[1]

    # 确定保存路径和文件名
    save_dir = os.path.join(output_root, category, str(digit))
    filename = f"{digit}_{sample_idx:04d}_{suffix}.png"
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # 随机选择字体
            font_path = random.choice(font_files)
            initial_size = 100
            
            # 获取字体和边界框
            font, bbox = _find_proper_font(font_path, str(digit), initial_size)
            if not font:
                continue
            base_padding = random.randint(-3, 0)
            # 计算图片尺寸
            extra_padding = 2 if config['morph'] else 0
            width = bbox[2] - bbox[0] + 2*(base_padding + extra_padding)
            height = bbox[3] - bbox[1] + 2*(base_padding + extra_padding)

            # 特殊处理数字1
            a = random.choice([1,1.5,2,2.5])
            if digit == 1:
                width = int(width * 2)

            # 创建画布
            img = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 计算绘制位置
            x = -bbox[0] + base_padding + extra_padding
            y = -bbox[1] + base_padding + extra_padding

            if digit == 1:
                x = width // 4 + 4*x
                
            draw.text((x, y), str(digit), font=font, fill=(0, 0, 0))
            
            # 形态学处理
            if config['morph']:
                morph_params = morph_config.get(config['morph'], {})
                img = apply_morphology(
                    img, 
                    operation=morph_params.get('operation', 'erode'),
                    kernel_size=morph_params.get('kernel_size', 3),
                    iterations=morph_params.get('iterations', 1)
                )
            
            # 添加下划线
            if config['underline']:
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

def apply_morphology(image, operation='erode', kernel_size=3, iterations=1):
    """应用形态学操作"""
    # 转换为灰度并反色
    gray = image.convert('L')
    inverted = Image.eval(gray, lambda x: 255 - x)
    
    # 应用多次操作
    for _ in range(iterations):
        if operation == 'dilate':
            inverted = inverted.filter(ImageFilter.MaxFilter(kernel_size))
        else:
            inverted = inverted.filter(ImageFilter.MinFilter(kernel_size))
    
    # 反色恢复并转回RGB
    result = Image.eval(inverted, lambda x: 255 - x).convert('RGB')
    return result

if __name__ == '__main__':
    generate_digits(
        font_dir='./nums/fonts',
        output_root='./nums/dataset',
        digits_range=(0, 10),
        total_samples=1000,
        base_padding=0,
        underline_config={'width': 5, 'padding': 8},
        morph_config={
            'thin': {'operation': 'erode', 'kernel_size': 3, 'iterations': 1}
        }
    )