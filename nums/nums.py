import os
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_all_fonts_samples(font_dir, output_root,
                              digits_range=(0, 10),
                              base_padding=2,
                              underline_config=None,
                              morph_config=None):
    """
    加载所有字体文件，为每种字体生成数字的不同版本
    
    修改要点：
    1. 只为6和9生成带下划线版本（UN和UT）
    2. 不为6和9生成无下划线版本
    3. 取消base/underlined文件夹分类
    4. 保存每一步腐蚀过程的中间图像
    """
    # 获取所有有效字体
    font_paths = _get_all_valid_fonts(font_dir)
    if not font_paths:
        raise ValueError(f"找不到有效字体文件：{font_dir}")
    
    print(f"找到 {len(font_paths)} 种有效字体")
    
    # 创建输出目录（单一结构）
    _create_dirs(output_root, digits_range)
    
    # 配置默认值
    underline_config = underline_config or {'width': 2, 'padding': 2}
    morph_config = morph_config or {
        'thin': {'operation': 'erode', 'kernel_size': 3, 'iterations': 1}
    }
    
    # 为每种字体生成数字
    for font_path in font_paths:
        font_name = _clean_font_name(os.path.basename(font_path))
        print(f"处理字体: {font_name}")
        
        # 检查字体是否支持数字
        if not _is_font_supports_numbers(font_path):
            print(f"字体 {font_name} 不支持数字，跳过")
            continue
            
        # 为每个数字生成合适版本
        for digit in range(*digits_range):
            if digit in (6, 9):
                # 只生成带下划线的细体版本 (UT)
                _generate_digit_image(
                    digit, 
                    font_path, 
                    output_root, 
                    base_padding,
                    'UT',
                    True,
                    'thin',
                    underline_config,
                    morph_config
                )
            else:
                # 只生成细体版本 (T) - 无下划线
                _generate_digit_image(
                    digit, 
                    font_path, 
                    output_root, 
                    base_padding,
                    'T',
                    False,
                    'thin',
                    underline_config,
                    morph_config
                )

def _get_all_valid_fonts(font_dir):
    """获取所有有效字体文件"""
    valid_fonts = []
    valid_ext = ('.ttf', '.otf')
    for f in os.listdir(font_dir):
        if f.lower().endswith(valid_ext):
            font_path = os.path.join(font_dir, f)
            if _is_valid_font(font_path):
                valid_fonts.append(font_path)
    return valid_fonts

def _is_font_supports_numbers(font_path):
    """检查字体是否支持数字"""
    try:
        font = ImageFont.truetype(font_path, 20)
        # 测试0-9是否都能渲染
        for digit in range(10):
            font.getbbox(str(digit))
        return True
    except Exception:
        return False

def _clean_font_name(font_name):
    """清理字体名称（移除特殊字符）"""
    return re.sub(r'[^a-zA-Z0-9-]', '_', os.path.splitext(font_name)[0])

def _is_valid_font(font_path):
    """验证字体有效性"""
    try:
        ImageFont.truetype(font_path, 10)
        return True
    except Exception:
        return False

def _create_dirs(output_root, digits_range):
    """创建输出目录结构（单一结构）"""
    for d in range(digits_range[0], digits_range[1]):
        os.makedirs(os.path.join(output_root, str(d)), exist_ok=True)

def _generate_digit_image(digit, font_path, output_root, base_padding,
                         suffix, add_underline, morph_type,
                         underline_config, morph_config):
    """生成单个数字的特定版本图像（包含每步腐蚀的保存）"""
    try:
        # 创建基本文件名
        font_name = _clean_font_name(os.path.basename(font_path))
        save_dir = os.path.join(output_root, str(digit))
        
        # 获取字体和边界框
        initial_size = 100
        font, bbox = _find_proper_font(font_path, str(digit), initial_size)
        if not font:
            print(f"字体不支持数字 {digit}")
            return
        
        # 计算图片尺寸
        extra_padding = 2 if morph_type else 0
        width = bbox[2] - bbox[0] + 2 * (base_padding + extra_padding)
        height = bbox[3] - bbox[1] + 2 * (base_padding + extra_padding)
        
        # 特殊处理数字1（加宽）
        if digit == 1:
            width = int(width * 2)
        
        # 创建画布
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 计算绘制位置
        x = -bbox[0] + base_padding + extra_padding
        y = -bbox[1] + base_padding + extra_padding
        
        if digit == 1:
            x = width // 4 + 4 * x
            
        draw.text((x, y), str(digit), font=font, fill=(0, 0, 0))
        
        # 添加下划线（仅针对6和9）
        if add_underline:
            img = _add_smart_underline(img, bbox, base_padding, underline_config)
        
        # 形态学处理 - 保存每一步的结果
        if morph_type:
            morph_params = morph_config.get(morph_type, {})
            iterations = morph_params.get('iterations', 1)
            operation = morph_params.get('operation', 'erode')
            kernel_size = morph_params.get('kernel_size', 3)
            
            # 保存原始图像（腐蚀前的状态）
            base_filename = f"{font_name}_{suffix}_step0.png"
            base_save_path = os.path.join(save_dir, base_filename)
            if not os.path.exists(base_save_path):
                img.save(base_save_path)
                print(f"保存腐蚀前图像: {base_save_path}")
            
            # 逐轮腐蚀并保存
            current_img = img.copy()
            for step in range(1, iterations + 1):
                # 应用一次腐蚀操作
                current_img = apply_morphology_single_step(
                    current_img, 
                    operation=operation,
                    kernel_size=kernel_size
                )
                
                # 保存腐蚀轮次图像
                step_filename = f"{font_name}_{suffix}_step{step}.png"
                step_save_path = os.path.join(save_dir, step_filename)
                
                if not os.path.exists(step_save_path):
                    current_img.save(step_save_path)
                    print(f"生成腐蚀轮次 {step}/{iterations}: {step_save_path}")
            
            # 保存最终版本（经过所有腐蚀轮次后的图像）
            final_filename = f"{font_name}_{suffix}.png"
            final_save_path = os.path.join(save_dir, final_filename)
            if not os.path.exists(final_save_path):
                current_img.save(final_save_path)
                print(f"生成最终: {final_save_path}")
        
        else:
            # 直接保存
            filename = f"{font_name}_{suffix}.png"
            save_path = os.path.join(save_dir, filename)
            
            if not os.path.exists(save_path):
                img.save(save_path)
                print(f"生成: {save_path}")
            
    except Exception as e:
        print(f"生成失败: 数字{digit} ({suffix}) - {str(e)}")

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

def apply_morphology_single_step(image, operation='erode', kernel_size=3):
    """
    应用单步形态学操作
    返回处理后的图像
    """
    # 转换为灰度并反色
    gray = image.convert('L')
    inverted = Image.eval(gray, lambda x: 255 - x)
    
    # 应用单次操作
    if operation == 'dilate':
        inverted = inverted.filter(ImageFilter.MaxFilter(kernel_size))
    else:  # erode
        inverted = inverted.filter(ImageFilter.MinFilter(kernel_size))
    
    # 反色恢复并转回RGB
    result = Image.eval(inverted, lambda x: 255 - x).convert('RGB')
    return result

if __name__ == '__main__':
    generate_all_fonts_samples(
        font_dir='./nums/fonts',
        output_root='../Datasets/nums_dataset1',
        digits_range=(0, 10),
        base_padding=0,
        underline_config={'width': 5, 'padding': 8},
        morph_config={
            'thin': {'operation': 'erode', 'kernel_size': 3, 'iterations': 3}
        }
    )