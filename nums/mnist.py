import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
from PIL import Image, ImageFilter
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 合并训练集和测试集
x_all = np.concatenate([x_train, x_test])
y_all = np.concatenate([y_train, y_test])

# 创建统一保存目录
base_dir = './mnist_images/all_classes'
os.makedirs(base_dir, exist_ok=True)

# 为每个数字创建对应的子文件夹
for i in range(10):
    os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)

def process_and_save(images, labels, target_dir):
    """
    处理并保存图像到对应数字的文件夹
    文件名格式：序号_标签.png (示例：123_7.png)
    """
    for idx, (image, label) in enumerate(zip(images, labels)):
        # 颜色反转（白底黑字）
        inverted = 255 - image
        
        # 转换为PIL图像
        img = Image.fromarray(inverted)
        
        # 转换为RGB三通道
        img = img.convert('RGB')
        
        # 高质量放大到96x96
        img = img.resize((96, 96), resample=Image.LANCZOS)
        
        # 锐化处理提高清晰度
        img = img.filter(ImageFilter.SHARPEN)
        
        # 构建保存路径
        save_path = os.path.join(target_dir, str(label), f"{idx}_{label}.png")
        img.save(save_path)

# 处理并保存全部图片
process_and_save(x_all, y_all, base_dir)

print(f"图片分类保存完成！共处理 {len(x_all)} 张图片。")
print(f"目录结构：\n{base_dir}")
print("├── 0\n├── 1\n├── 2\n...\n└── 9")