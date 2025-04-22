import os
from PIL import Image
import argparse

def process_image(input_path, output_path, target_size):
    """
    处理单个图片的缩放并保存
    """
    with Image.open(input_path) as img:
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        if img.format:
            resized_img.save(output_path, format=img.format)
        else:
            resized_img.save(output_path)

def process_directory(input_dir, output_dir, target_size):
    """
    递归处理目录及其子目录中的所有图片（添加_resized后缀防覆盖）
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(supported_formats):
                # 分割文件名与扩展名
                filename, ext = os.path.splitext(file)
                # 添加_resized后缀
                new_filename = f"{filename}_resized{ext}"
                
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_path, new_filename)  # 使用新文件名
                
                try:
                    process_image(input_file, output_file, target_size)
                    print(f"Processed: {input_file} -> {output_file}")
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量缩放图片工具（防覆盖版）')
    parser.add_argument('--input', default='./nums/dataset', help='输入目录路径')
    parser.add_argument('--output', default='./nums/dataset', help='输出目录路径')
    parser.add_argument('--width', type=int, default=96, help='目标宽度')
    parser.add_argument('--height', type=int, default=96, help='目标高度')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    process_directory(args.input, args.output, (args.width, args.height))