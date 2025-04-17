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
        ImageFont.truetype(font_path, 10)
        return True
    except Exception:
        return False

def _generate_version(digit, fonts, output_root,
                     sample_idx, image_size,
                     scale_factor, h_scale_range,
                     v_scale_range, underline=False,
                     underline_width=None):
    """
    生成单个版本图片
    """
    # 确定版本参数
    version = 'underlined' if underline else 'normal'
    suffix = '_u' if underline else ''
    save_dir = os.path.join(output_root, version, str(digit))
    
    # 生成参数
    base_font_size = int(image_size[1] * scale_factor)
    max_attempts = 5  # 更严格的尝试次数限制

    for _ in range(max_attempts):
        try:
            # 随机选择字体
            font_path = random.choice(fonts)
            font = ImageFont.truetype(font_path, base_font_size)
            
            # 创建画布
            img = Image.new('RGB', image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 动态缩放
            h_scale = random.uniform(*h_scale_range)
            v_scale = random.uniform(*v_scale_range)
            temp_size = (int(image_size[0] * h_scale), int(image_size[1] * v_scale))

            # 临时绘图层
            temp_img = Image.new('RGBA', temp_size, (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # 文字定位
            text_pos = (temp_size[0]//2, temp_size[1]//2)
            temp_draw.text(text_pos, str(digit), font=font, fill=(0, 0, 0, 255), anchor='mm')

            # 下划线处理
            if underline:
                ascent, descent = font.getmetrics()
                text_bbox = font.getbbox(str(digit))
                text_width = text_bbox[2] - text_bbox[0]
                
                # 智能下划线参数计算
                line_width = underline_width or max(2, int((ascent + descent) * 0.1))
                radius = line_width // 2
                underline_height = line_width + 2 * radius
                
                # 下划线定位
                line_y = temp_size[1]//2 + (ascent - descent)//2 + int(descent * 0.25)
                underline_pos = (
                    temp_size[0]//2 - text_width//2 - radius,
                    line_y - radius
                )

                # 绘制下划线
                underline_img = Image.new('RGBA', (text_width + 2*radius, underline_height), (0, 0, 0, 0))
                underline_draw = ImageDraw.Draw(underline_img)
                underline_draw.rounded_rectangle(
                    (0, 0, text_width + 2*radius - 1, underline_height - 1),
                    radius=radius,
                    fill=(0, 0, 0, 255)
                )
                temp_img.alpha_composite(underline_img, underline_pos)

            # 随机偏移合成
            max_x = max(0, (image_size[0] - temp_size[0]) // 2)
            max_y = max(0, (image_size[1] - temp_size[1]) // 2)
            pos = (max_x + random.randint(-2, 2), max_y + random.randint(-2, 2))
            img.paste(temp_img, pos, temp_img)

            # 保存文件
            save_path = os.path.join(save_dir, f"{digit}_{sample_idx:04d}{suffix}.png")
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
        output_root='./',
        digits_range=(0, 99),
        total_samples=99,
        image_size=(96, 96),
        scale_factor=0.8,
        h_scale_range=(0.9, 1.1),
        v_scale_range=(0.9, 1.1),
        underline_width=3
    )