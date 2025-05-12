import os
import random
from PIL import Image

def generate_two_digits(input_root, output_root, total_samples=10000, spacing=0):
    """
    合成两位数的图片，确保每个组合均匀分布
    
    参数说明：
    - input_root: 个位数图片的根目录
    - output_root: 输出根目录
    - total_samples: 总样本数量
    - spacing: 数字间水平间距（像素）
    """
    # 计算每个两位数的样本数
    samples_per_two_digit, remainder = divmod(total_samples, 100)
    samples_distribution = [samples_per_two_digit + 1 if i < remainder else samples_per_two_digit for i in range(100)]

    # 收集个位数图片路径
    versions = ['', '']
    digit_paths = {v: {d: [] for d in range(10)} for v in versions}
    
    for version in versions:
        for digit in range(10):
            dir_path = os.path.join(input_root, version, str(digit))
            if os.path.exists(dir_path):
                digit_paths[version][digit] = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.endswith('.png')
                ]

    # 创建输出目录
    for version in versions:
        for two_digit in range(100):
            output_dir = os.path.join(output_root, version, 'two_digits', f"{two_digit:02d}")
            os.makedirs(output_dir, exist_ok=True)

    # 生成图片
    for version in versions:
        print(f"\n正在生成 {version} 版本...")
        for two_digit in range(100):
            d1, d2 = two_digit // 10, two_digit % 10
            samples_needed = samples_distribution[two_digit]
            
            # 获取可用路径
            d1_paths = digit_paths[version][d1]
            d2_paths = digit_paths[version][d2]
            
            if not d1_paths or not d2_paths:
                print(f"跳过 {two_digit:02d}：缺少数字素材")
                continue
                
            print(f"生成 {two_digit:02d}（需要{samples_needed}个样本）")
            
            # 确保有足够的样本组合
            for i in range(samples_needed):
                # 随机选择不重复的组合（简单实现）
                img1 = random.choice(d1_paths)
                img2 = random.choice(d2_paths)
                
                # 合成并保存
                combined = combine_digits(img1, img2, spacing)
                output_path = os.path.join(
                    output_root,
                    version,
                    'two_digits',
                    f"{two_digit:02d}",
                    f"{two_digit:02d}_{i:04d}.png"
                )
                combined.save(output_path)

def combine_digits(img_path1, img_path2, spacing):
    """合成两个数字图片，保持垂直居中"""
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    
    w1, h1 = img1.size
    w2, h2 = img2.size
    spacing = random.randint(-3, 0)
    # 计算新图片尺寸
    total_width = w1 + w2 + spacing
    max_height = max(h1, h2)
    
    new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # 计算垂直偏移
    y_offset1 = (max_height - h1) // 2
    y_offset2 = (max_height - h2) // 2
    
    new_img.paste(img1, (0, y_offset1))
    new_img.paste(img2, (w1 + spacing, y_offset2))
    
    return new_img

if __name__ == '__main__':
    # 先生成个位数字
    # generate_digits(
    #     font_dir='./nums/fonts',
    #     output_root='./nums/dataset',
    #     digits_range=(0, 10),
    #     total_samples=1000,  # 每个数字生成100个样本
    #     base_padding=2,
    #     underline_config={
    #         'width': 4,
    #         'padding': 4
    #     }
    # )
    
    # 再合成两位数
    generate_two_digits(
        input_root='./nums/dataset',
        output_root='./nums/dataset3',
        spacing = 0,
        total_samples=200  
    )