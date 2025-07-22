import os
import random
from PIL import Image

def generate_digit_images(parent_dir, output_dir, num_images=100, spacing=0):
    """
    生成指定数量的两位数组合图片，并为每个两位数创建单独的子文件夹
    :param parent_dir: 包含0-9数字子文件夹的父目录
    :param output_dir: 合成图片的输出根目录
    :param num_images: 要生成的图片总数（默认100张）
    :param spacing: 个位和十位数字之间的间距（像素，默认0）
    """
    # 确保输出根目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 预加载每个数字的图片路径
    digit_images = {str(i): [] for i in range(10)}
    
    # 扫描数字文件夹 (0-9)
    for digit in digit_images.keys():
        folder_path = os.path.join(parent_dir, digit)
        if not os.path.exists(folder_path):
            print(f"警告: 数字文件夹 {digit} 不存在: {folder_path}")
            continue
            
        # 收集文件夹中的图片文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                digit_images[digit].append(file_path)
    
    # 验证是否有足够的图片
    for digit in digit_images:
        if not digit_images[digit]:
            print(f"错误: 数字文件夹 {digit} 中没有图片")
            return
    
    # 生成指定数量的两位数图片
    for i in range(num_images):
        # 随机选择两位数（00-99）
        tens = random.randint(0, 9)
        ones = random.randint(0, 9)
        target_num = f"{tens}{ones}"
        ten_str = str(tens)
        one_str = str(ones)
        
        # 创建目标输出文件夹（以两位数命名）
        num_output_dir = os.path.join(output_dir, target_num)
        os.makedirs(num_output_dir, exist_ok=True)
        
        # 随机选择图片
        tens_image_path = random.choice(digit_images[ten_str])
        ones_image_path = random.choice(digit_images[one_str])
        
        try:
            # 加载并合成图片
            with Image.open(tens_image_path) as img_tens, \
                 Image.open(ones_image_path) as img_ones:
                
                # 统一图片高度
                height = max(img_tens.height, img_ones.height)
                
                # 调整图片尺寸（保持比例）
                def resize_with_aspect(img, target_height):
                    ratio = target_height / img.height
                    new_width = int(img.width * ratio)
                    return img.resize((new_width, target_height), Image.LANCZOS)
                
                img_tens = resize_with_aspect(img_tens, height)
                img_ones = resize_with_aspect(img_ones, height)
                
                # 创建新画布（添加间距）
                spacing =  int(random.uniform(-5, 5))

                combined_width = img_tens.width + img_ones.width + spacing
                combined_img = Image.new('RGB', (combined_width, height), color=(255, 255, 255))
                
                # 粘贴图片（添加间距）
                combined_img.paste(img_tens, (0, 0))
                combined_img.paste(img_ones, (img_tens.width + spacing, 0))
                
                # 生成唯一文件名（避免覆盖）
                file_count = len(os.listdir(num_output_dir))
                output_path = os.path.join(num_output_dir, f"{target_num}_{file_count+1}.jpg")
                
                # 保存结果
                combined_img.save(output_path, quality=95)
                print(f"已生成: {output_path}")
                
        except Exception as e:
            print(f"生成 {target_num} 时出错: {str(e)}")

if __name__ == "__main__":
    # 根据截图1的实际路径修改
    input_root = "../Datasets/9_dataset_3"  # 包含0-9文件夹的根目录
    
    # 输出根目录（根据截图2修改）
    output_root = "../Datasets/99_dataset"   # 合成图片的输出根目录
    
    # 自定义参数
    num_images = 100000  # 要生成的图片总数
    spacing = 0      # 个位和十位之间的间距（像素）
    
    generate_digit_images(input_root, output_root, num_images, spacing)