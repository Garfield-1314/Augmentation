import os
import random
from PIL import Image, ImageDraw, ImageFont

def generate_custom_digits(font_dir, output_root,
                          digits_range=(0, 9),
                          total_samples=1000,
                          image_size=(64, 64),
                          scale_factor=0.7,
                          h_scale_range=(0.8, 1.2),
                          v_scale_range=(0.8, 1.2),
                          underline_width=None):
    """
    自定义数字图片生成器（双版本严格分离）
    
    参数说明：
    - output_root/
      ├── normal/         # 普通版本目录
      │   └── [0-9]/     # 数字子目录
      └── underlined/    # 下划线版本目录
          └── [0-9]/     # 数字子目录
    """
    font_files = _get_valid_fonts(font_dir)
    if not font_files:
        raise ValueError(f"未找到有效字体文件：{font_dir}")

    # 创建双版本目录结构
    for version in ['normal', 'underlined']:
        version_dir = os.path.join(output_root, version)
        for d in range(digits_range[0], digits_range[1] + 1):
            os.makedirs(os.path.join(version_dir, str(d)), exist_ok=True)

    # 样本分配逻辑
    num_digits = digits_range[1] - digits_range[0] + 1
    samples_per_digit = total_samples // num_digits
    remainder = total_samples % num_digits

    # 双版本生成逻辑
    for idx, digit in enumerate(range(digits_range[0], digits_range[1] + 1)):
        current_samples = samples_per_digit + (1 if idx < remainder else 0)
        for sample_idx in range(current_samples):
            # 生成普通版本
            _generate_version(
                digit=digit,
                fonts=font_files.copy(),
                output_root=output_root,
                sample_idx=sample_idx,
                image_size=image_size,
                scale_factor=scale_factor,
                h_scale_range=h_scale_range,
                v_scale_range=v_scale_range,
                underline=False,
                underline_width=underline_width
            )
            
            # 生成下划线版本
            _generate_version(
                digit=digit,
                fonts=font_files.copy(),
                output_root=output_root,
                sample_idx=sample_idx,
                image_size=image_size,
                scale_factor=scale_factor,
                h_scale_range=h_scale_range,
                v_scale_range=v_scale_range,
                underline=True,
                underline_width=underline_width
            )

def _get_valid_fonts(font_dir):
    """获取有效字体文件列表"""
    valid_ext = ('.ttf', '.otf', '.ttc')
    return [
        os.path.join(font_dir, f) 
        for f in os.listdir(font_dir) 
        if f.lower().endswith(valid_ext) and _is_valid_font(os.path.join(font_dir, f))
    ]

def _is_valid_font(font_path):
    """验证字体有效性"""
    try:
        font = ImageFont.truetype(font_path, 10)
        # 额外检查是否包含数字字符
        font.getbbox('0')
        return True
    except Exception:
        return False

def _generate_version(digit, fonts, output_root,
                     sample_idx, image_size,
                     scale_factor, h_scale_range,
                     v_scale_range, underline=False,
                     underline_width=None):
    """
    生成单个版本图片（支持多字符下划线）
    """
    version = 'underlined' if underline else 'normal'
    suffix = '_u' if underline else ''
    save_dir = os.path.join(output_root, version, str(digit))
    
    base_font_size = int(image_size[1] * scale_factor)
    max_attempts = 5

    for _ in range(max_attempts):
        try:
            font_path = random.choice(fonts)
            font = ImageFont.truetype(font_path, base_font_size)
            
            # 创建主画布
            img = Image.new('RGB', image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 动态缩放
            h_scale = random.uniform(*h_scale_range)
            v_scale = random.uniform(*v_scale_range)
            temp_size = (
                int(image_size[0] * h_scale),
                int(image_size[1] * v_scale)
            )

            # 创建临时绘图层
            temp_img = Image.new('RGBA', temp_size, (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # 解析数字字符串
            s = str(digit)
            chars = list(s)
            
            # 获取字体参数
            ascent, descent = font.getmetrics()
            font_height = ascent + descent
            
            # 计算字符布局
            char_widths = []
            spacings = []
            for c in chars:
                bbox = font.getbbox(c)
                char_widths.append(bbox[2] - bbox[0])
                spacings.append(font.getlength(c) * 0.1)  # 10%字符宽度作为间距
            
            total_width = sum(w + s for w, s in zip(char_widths, spacings)) - spacings[-1]
            start_x = (temp_size[0] - total_width) // 2
            current_x = start_x
            
            # 绘制字符
            for c, c_width, spacing in zip(chars, char_widths, spacings):
                # 计算字符位置
                text_y = (temp_size[1] - font_height) // 2 + ascent
                
                # 绘制字符
                temp_draw.text(
                    (current_x, text_y),
                    c,
                    font=font,
                    fill=(0, 0, 0, 255),
                    anchor='ls'  # 左基线对齐
                )
                
                # 绘制下划线
                if underline:
                    line_width = underline_width or max(2, int(font_height * 0.1))
                    underline_y = text_y + descent // 2 - 10
                    
                    # 单个字符下划线
                    underline_bbox = (
                        current_x - line_width//2,
                        underline_y,
                        current_x + c_width + line_width//2,
                        underline_y + line_width
                    )
                    temp_draw.rectangle(
                        underline_bbox,
                        fill=(0, 0, 0, 255)
                    )
                
                current_x += c_width + spacing

            # 随机偏移合成
            max_offset_x = max(0, (image_size[0] - temp_size[0]) // 2)
            max_offset_y = max(0, (image_size[1] - temp_size[1]) // 2)
            pos = (
                max_offset_x + random.randint(-2, 2),
                max_offset_y + random.randint(-2, 2)
            )
            
            img.paste(temp_img, pos, temp_img)

            # 保存文件
            save_path = os.path.join(save_dir, f"{digit}_{sample_idx:04d}{suffix}.jpg")
            img.save(save_path)
            return

        except Exception as e:
            if font_path in fonts:
                fonts.remove(font_path)
            if not fonts:
                raise RuntimeError("所有字体均无效")

    raise RuntimeError(f"生成失败：数字{digit} 样本{sample_idx}")

if __name__ == '__main__':
    generate_custom_digits(
        font_dir='./fonts',
        output_root='./dataset',
        digits_range=(0, 99),
        total_samples=100,
        image_size=(96, 96),
        scale_factor=0.8,  # 减小字体比例确保显示
        h_scale_range=(0.9, 1.0),  # 缩小缩放范围
        v_scale_range=(0.9, 1.0),
        underline_width=3
    )