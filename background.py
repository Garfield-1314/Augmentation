import numpy as np
import matplotlib.pyplot as plt
import os

def generate_noise_image(size=(224, 224), save_path=None, white_background=False):
    """
    生成随机噪声图像或纯白色背景图像
    
    参数：
    size (tuple): 图像尺寸，默认(128, 128)
    save_path (str): 保存路径，默认不保存
    white_background (bool): 是否使用纯白背景，默认False生成随机噪声
    """
    # 根据参数选择背景类型
    if white_background:
        # 创建纯白色背景（RGB三通道均为1）
        noise = np.ones((size[0], size[1], 3))
    else:
        # 生成随机噪声（范围0-1）
        noise = np.random.rand(size[0], size[1], 3)
    
    # 显示噪声图片
    plt.imshow(noise)
    plt.axis('off')  # 隐藏坐标轴
    
    # 保存逻辑保持不变
    if save_path:
        folder_path = os.path.dirname(save_path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.imsave(save_path, noise)
    
    plt.show()

# 示例用法
# 生成普通噪声图（默认）
# generate_noise_image(save_path='./dataset/background/noise_image.png')

# 生成白色背景图
generate_noise_image(
    save_path='./background/white_image.png',
    white_background=True
)