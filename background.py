import numpy as np
import matplotlib.pyplot as plt
import os

def generate_noise_image(size=(224, 224), save_path=None, white_background=False, noise_density=0.05):
    """
    生成随机噪声图像或纯白色背景图像（可添加黑色噪点）
    
    参数：
    size (tuple): 图像尺寸，默认(135, 135)
    save_path (str): 保存路径，默认不保存
    white_background (bool): 是否使用纯白背景，默认False生成随机噪声
    noise_density (float): 黑色噪点密度（0-1），默认0.05（5%的像素为噪点）
    """
    # 根据参数选择背景类型
    if white_background:
        # 创建纯白色背景（RGB三通道均为1）
        img = np.ones((size[0], size[1], 3))
        
        # 添加黑色噪点
        if noise_density > 0:
            # 创建二值掩码（1表示保留白色，0表示变为黑点）
            mask = np.random.choice([0, 1], 
                                   size=(size[0], size[1], 1), 
                                   p=[noise_density, 1-noise_density])
            # 将掩码扩展到RGB通道
            mask = np.repeat(mask, 3, axis=2)
            # 应用掩码：0的位置变为黑色，1的位置保持白色
            img = img * mask
    else:
        # 生成随机噪声（范围0-1）
        img = np.random.rand(size[0], size[1], 3)
    
    # 显示图片
    plt.imshow(img)
    plt.axis('off')  # 隐藏坐标轴
    
    # 保存逻辑
    if save_path:
        folder_path = os.path.dirname(save_path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.imsave(save_path, img)
    
    plt.show()

# 示例：生成带黑色噪点的白背景
generate_noise_image(
    size=(135, 135),              # 200x200像素
    save_path='../Datasets/background_w_135/noisy_white_bg.png', # 保存路径
    white_background=True,         # 使用白色背景
    noise_density=0.00             # 3%像素为黑色噪点
)