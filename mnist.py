
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
from PIL import Image

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 指定保存的根目录
base_dir = './mnist_images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 为每个类别创建训练和测试文件夹
for dir_path in [train_dir, test_dir]:
    for i in range(10):
        os.makedirs(os.path.join(dir_path, str(i)), exist_ok=True)

def save_images(images, labels, target_dir):
    """
    将图像和标签保存到目标目录下的对应子文件夹
    """
    for idx, (image, label) in enumerate(zip(images, labels)):
        # 构造保存路径：目标目录/标签/索引.png
        folder = os.path.join(target_dir, str(label))
        filename = os.path.join(folder, f"{idx}.png")
        # 将numpy数组转换为PIL图像并保存
        Image.fromarray(image).save(filename)

# 处理训练集和测试集
save_images(x_train, y_train, train_dir)
save_images(x_test, y_test, test_dir)

print("图片保存完成！")