import numpy as np
import matplotlib.pyplot as plt
import os

def generate_noise_image(size=(224, 224), save_path=None, white_background=False, noise_density=0.05, show=False):
    """
    生成随机噪声图像或纯白色背景图像（可添加黑色噪点）
    
    参数：
    size (tuple): 图像尺寸 (height, width)
    save_path (str): 保存路径，默认不保存
    white_background (bool): 是否使用纯白背景，默认False生成彩色随机噪声
    noise_density (float): 黑色噪点密度（0-1）
    show (bool): 是否使用plt弹出显示
    """
    # 根据参数选择背景类型
    if white_background:
        # 创建纯白色背景
        img = np.ones((size[0], size[1], 3))
        
        # 添加黑色噪点
        if noise_density > 0:
            mask = np.random.choice([0, 1], 
                                   size=(size[0], size[1], 1), 
                                   p=[noise_density, 1-noise_density])
            img = img * np.repeat(mask, 3, axis=2)
    else:
        # 生成彩色随机噪声
        img = np.random.rand(size[0], size[1], 3)
    
    # 保存逻辑
    if save_path:
        print(f"Generating background images to {save_path}...")
        folder_path = os.path.dirname(save_path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.imsave(save_path, img)
    
    # 显示逻辑
    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    return img

if __name__ == "__main__":
    # 默认调用示例
    generate_noise_image(
        size=(160, 240),
        save_path='../Datasets/background/noisy_white_bg.png',
        white_background=True,
        noise_density=0.00,
        show=True
    )
