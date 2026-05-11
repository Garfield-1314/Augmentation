import os
import cv2
import numpy as np
import datetime
from PIL import Image
from tqdm import tqdm

# --- 核心图像处理函数 ---

def Scale(image, scale):
    """缩放"""
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def Horizontal(image):
    """水平翻转"""
    return cv2.flip(image, 1, dst=None)

def Vertical(image):
    """垂直翻转"""
    return cv2.flip(image, 0, dst=None)

def Rotate(image, angle, scale=1.0):
    """旋转"""
    w = image.shape[1]
    h = image.shape[0]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    image = cv2.warpAffine(image, M, (w, h), borderValue=(255, 0, 0))
    return image

def Move(img, x, y):
    """平移"""
    img_info = img.shape
    height = img_info[0]
    width = img_info[1]
    mat_translation = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(img, mat_translation, (width, height))
    return dst

def SaltAndPepper(src, percetage=0.01):
    """椒盐噪声"""
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0]-1)
        randG = np.random.randint(0, src.shape[1]-1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg

def GaussianNoise(image, percetage=0.01):
    """高斯噪声"""
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def Blur(img):
    """模糊"""
    return cv2.GaussianBlur(img, (3, 3), 1)

def compress_img_CV(img, target_width=800, target_height=600, mode='stretch'):
    """图像缩放/压缩
    :param mode: 'stretch' (整体拉伸) 或 'crop' (中心裁剪)
    """
    if mode == 'stretch':
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    elif mode == 'crop':
        h, w = img.shape[:2]
        # 计算比例
        scale_w = target_width / w
        scale_h = target_height / h
        scale = max(scale_w, scale_h)
        
        # 先按比例缩放，使得一边对齐，另一边超出
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 中心裁剪
        start_x = (new_w - target_width) // 2
        start_y = (new_h - target_height) // 2
        return resized[start_y:start_y+target_height, start_x:start_x+target_width]
    else:
        raise ValueError("Mode must be 'stretch' or 'crop'")

def Darker_Brighter(image, percetage):
    """明暗调节"""
    return cv2.multiply(image, percetage)

def Contrast(image, percetage):
    """对比度调节"""
    return cv2.convertScaleAbs(image, alpha=percetage, beta=0)

def hsv(image, percetage):
    """饱和度调节"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], percetage)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def hue(image, percetage):
    """色调调节"""
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_image[:, :, 0] = (hue_image[:, :, 0] + percetage) % 180
    return cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)

def pixelate(image, pixel_size):
    """像素化处理"""
    height, width, _ = image.shape
    new_width = width // pixel_size * pixel_size
    new_height = height // pixel_size * pixel_size
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    pixelated_image = resized_image.copy()
    for i in range(0, new_width, pixel_size):
        for j in range(0, new_height, pixel_size):
            block = resized_image[j:j+pixel_size, i:i+pixel_size]
            average_color = block.mean(axis=0).mean(axis=0)
            pixelated_image[j:j+pixel_size, i:i+pixel_size] = average_color
    return cv2.resize(pixelated_image, (width, height), interpolation=cv2.INTER_NEAREST)

def make_square(image):
    """填充为正方形"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = img_pil.size
    new_size = max(width, height)
    new_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(img_pil, paste_position)
    return cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)


def LocalIllumination(image, num_spots=None, strength_range=(-150, 200), radius_range=None):
    """
    模拟局部光照不均（包括局部过曝和局部阴影/过暗）
    :param image: 输入图像 (BGR 格式)
    :param num_spots: 光照/阴影点数量，默认随机 1-3 个
    :param strength_range: 亮度变化范围。正数为增亮，负数为变暗
    :param radius_range: 影响半径范围，默认根据图像尺寸自动计算
    :return: 增强后的图像
    """
    h, w, c = image.shape
    if num_spots is None:
        num_spots = np.random.randint(1, 4)
    
    if radius_range is None:
        min_dim = min(h, w)
        radius_range = (min_dim // 8, min_dim // 3)
    
    result = image.astype(np.float32)
    y, x = np.ogrid[:h, :w]
    
    for _ in range(num_spots):
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        # 随机选择强度，可能为正（光亮）或负（阴影）
        strength = np.random.randint(strength_range[0], strength_range[1])
        radius = np.random.randint(radius_range[0], radius_range[1])
        
        sigma = radius / 2.0
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        mask = np.exp(-dist_sq / (2 * sigma**2))
        
        # 叠加亮度变化（加法实现增亮，减法实现变暗）
        result += mask[:, :, np.newaxis] * strength
        
    return np.clip(result, 0, 255).astype(np.uint8)

# --- 批量处理封装 ---

def process_directory(rootpath, savepath, process_func, suffix, **kwargs):
    """通用目录遍历处理函数"""
    for a, b, c in os.walk(rootpath):
        relative_path = os.path.relpath(a, rootpath)
        current_save_path = os.path.join(savepath, relative_path)
        
        for file_i in tqdm(c, desc=f"Processing {relative_path}"):
            if not file_i.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            file_i_path = os.path.join(a, file_i)
            img_i = cv2.imread(file_i_path)
            if img_i is None: continue

            os.makedirs(current_save_path, exist_ok=True)
            
            result = process_func(img_i, **kwargs)
            
            base_name, ext = os.path.splitext(file_i)
            save_name = f"{base_name}_{suffix}{ext}"
            cv2.imwrite(os.path.join(current_save_path, save_name), result)

# 具体的批量调用封装
def batch_yasuo(root, save, w=800, h=600, mode='stretch'):
    print(f"Compressing images to {w}x{h} with mode {mode}...")
    process_directory(root, save, compress_img_CV, f"{w}x{h}_{mode}", target_width=w, target_height=h, mode=mode)

def batch_rotate(root, save, angles=[90, 180, 270]):
    for angle in angles:
        process_directory(root, save, Rotate, f"rotate{angle}", angle=angle)

def batch_pixelate(root, save, size=10):
    print(f"Pixelating images with size {size}...")
    process_directory(root, save, pixelate, f"pixelated", pixel_size=size)

def batch_local_illumination(root, save, strength_range=(-150, 200), radius_range=None, num_spots=None):
    print("Adding local illumination augmentation...")
    process_directory(root, save, LocalIllumination, f"illumination", strength_range=strength_range, radius_range=radius_range, num_spots=num_spots)

def batch_flip(root, save):
    process_directory(root, save, Horizontal, "Hor")
    process_directory(root, save, Vertical, "Ver")

def batch_noise(root, save):
    process_directory(root, save, GaussianNoise, "Gauss", percetage=0.01)
    process_directory(root, save, SaltAndPepper, "Salt", percetage=0.01)

def batch_brightness(root, save):
    process_directory(root, save, Darker_Brighter, "brighter", percetage=1.5)
    process_directory(root, save, Darker_Brighter, "darker", percetage=0.75)

if __name__ == "__main__":
    # 示例运行逻辑
    ROOT = '../smartcar26'
    SAVE = '../Datasets/smartcar26_160'
    batch_yasuo(ROOT, SAVE, 160, 160)
    
    ROOT_PIXEL = '../Datasets/smartcar26_160'
    SAVE_PIXEL = '../Datasets/smartcar26_160_pixelated'
    batch_pixelate(ROOT_PIXEL, SAVE_PIXEL, size=3)

 