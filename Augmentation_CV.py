'''
    这是图片数据增强的代码，可以对图片实现：
    1. 尺寸放大缩小
    2. 旋转（任意角度，如45°，90°，180°，270°）
    3. 翻转（水平翻转，垂直翻转）
    4. 明亮度改变（变亮，变暗）
    5. 像素平移（往一个方向平移像素，空出部分自动填补黑色）
    6. 添加噪声（椒盐噪声，高斯噪声）
'''
import os
import cv2
import numpy as np
from PIL import Image
'''
缩放
'''
# 放大缩小
def Scale(image, scale):
    return cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)

####################翻转###########################################
####################翻转###########################################
####################翻转###########################################
# 水平翻转
def Horizontal(image):
    return cv2.flip(image,1,dst=None) #水平镜像

# 垂直翻转
def Vertical(image):
    return cv2.flip(image,0,dst=None) #垂直镜像

def Horizontal_Vertical(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)

            img_Hor = Horizontal(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_Hor.jpg"), img_Hor)

            img_Ver = Vertical(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_Ver.jpg"), img_Ver)

####################翻转###########################################
####################翻转###########################################
####################翻转###########################################

####################旋转###########################################
####################旋转###########################################
####################旋转###########################################
def Rotate(image, angle, scale):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h),borderValue=(255,0,0))
    return image

def Rotate_45(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
            
            img_rotate = Rotate(img_i, 45,0.6)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_x.jpg"), img_rotate)


def Rotate_90_180_270(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)
            
            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
            
            img_rotate = Rotate(img_i, 90,1)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_90.jpg"), img_rotate)
            
            img_rotate = Rotate(img_i, 180,1)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_180.jpg"), img_rotate)

            img_rotate = Rotate(img_i, 270,1)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_270.jpg"), img_rotate)
####################旋转###########################################
####################旋转###########################################
####################旋转###########################################


####################平移###########################################
####################平移###########################################
####################平移###########################################
def Move(img,x,y):
    img_info=img.shape
    height=img_info[0]
    width=img_info[1]

    mat_translation=np.float32([[1,0,x],[0,1,y]])  #变换矩阵：设置平移变换所需的计算矩阵：2行3列
    #[[1,0,20],[0,1,50]]   表示平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离。
    dst=cv2.warpAffine(img,mat_translation,(width,height))  #变换函数
    return dst

def move_img(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)   

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
            
            img_rotate = Move(img_i, 20,20)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_move.jpg"), img_rotate)   
####################平移###########################################
####################平移###########################################
####################平移###########################################

####################椒盐、高斯噪声###########################################
####################椒盐、高斯噪声###########################################
####################椒盐、高斯噪声###########################################
def SaltAndPepper(src,percetage=0.01):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg

# 高斯噪声
def GaussianNoise(image,percetage=0.01):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def G_and_S(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)

            img_Gauss = GaussianNoise(img_i,0.01)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_Gauss.jpg"),img_Gauss)

            img_Salt = SaltAndPepper(img_i,0.01)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_Salt.jpg"),img_Salt)

####################椒盐、高斯噪声###########################################
####################椒盐、高斯噪声###########################################
####################椒盐、高斯噪声###########################################

####################模糊图片###########################################
####################模糊图片###########################################
####################模糊图片###########################################
def Blur(img):
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    return blur
####################模糊图片###########################################
####################模糊图片###########################################
####################模糊图片###########################################

####################压缩图片###########################################
####################压缩图片###########################################
####################压缩图片###########################################
def compress_img_CV(img, target_width=800, target_height=600):
    """
    将图像缩放到固定大小
    :param img: 输入图像
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 缩放后的图像
    """
    img_resize = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return img_resize

def YASUO(rootpath, savepath, target_width=800, target_height=600):
    """
    遍历指定目录下的所有图片文件，将其缩放到固定大小并保存到目标目录
    :param rootpath: 源图片目录
    :param savepath: 保存目录
    :param target_width: 目标宽度
    :param target_height: 目标高度
    """
    save_loc = savepath
    for a, b, c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(f"处理文件: {file_i_path}")
            
            # 获取文件所在目录名
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc, dir_loc)

            # 如果保存路径不存在，则创建
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            # 读取图像
            img_i = cv2.imread(file_i_path)
            if img_i is None:
                print(f"无法读取文件: {file_i_path}")
                continue

            # 缩放图像
            img_yasuo = compress_img_CV(img_i, target_width=target_width, target_height=target_height)

            # 保存缩放后的图像
            base_name = os.path.splitext(file_i)[0]  # 获取文件名（不含扩展名）
            save_file_name = f"{base_name}_80.jpg"
            save_file_path = os.path.join(save_path, save_file_name)
            cv2.imwrite(save_file_path, img_yasuo)
            # print(f"图像已保存到: {save_file_path}")
####################压缩图片###########################################
####################压缩图片###########################################
####################压缩图片###########################################

####################亮暗###########################################
####################亮暗###########################################
####################亮暗###########################################
def Darker_Brighter(image,percetage):
    brightness_factor = percetage  # 亮度增强系数
    brightened_image = cv2.multiply(image, brightness_factor)
    return brightened_image

def D_dan_B(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)

            img_dar = Darker_Brighter(img_i,1.5)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(1.25) + "_bar.jpg"), img_dar)

          
            img_bar = Darker_Brighter(img_i,0.75)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(0.75) + "_dar.jpg"), img_bar)                        
####################亮暗###########################################
####################亮暗###########################################
####################亮暗###########################################

####################对比度###########################################
####################对比度###########################################
####################对比度###########################################
def Contrast(image,percetage):
    contrast_factor = percetage
    contrasted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return contrasted_image

def Contrast_image(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
            
            i=0.3
            img__Contrastd = Contrast(img_i,1-i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_Contrastd.jpg"), img__Contrastd)

            img__Contrasth = Contrast(img_i,1+i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_Contrasth.jpg"), img__Contrasth)                        
####################对比度###########################################
####################对比度###########################################
####################对比度###########################################

####################饱和度###########################################
####################饱和度###########################################
####################饱和度###########################################
def hsv(image,percetage):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_factor = percetage # 饱和度增强系数
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return saturated_image

def hsv_image(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
            
            i=0.25
            img_hsvd = hsv(img_i,1-i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_hsvd.jpg"), img_hsvd)

            img_hsvh = hsv(img_i,1+i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_hsvh.jpg"), img_hsvh)                        
####################饱和度###########################################
####################饱和度###########################################
####################饱和度###########################################

####################色调扰动###########################################
####################色调扰动###########################################
####################色调扰动###########################################
def hue(image,percetage):
    hue_shift = percetage  # 色调偏移量
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_image[:, :, 0] = (hue_image[:, :, 0] + hue_shift) % 180
    hue_shifted_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
    return hue_shifted_image

def hue_image(rootpath,savepath):
    save_loc = savepath
    for a,b,c in os.walk(rootpath):
        for file_i in c:
            file_i_path = os.path.join(a,file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            dir_loc = os.path.split(split[0])[1]
            save_path = os.path.join(save_loc,dir_loc)

            if not os.path.exists(save_path):  # 先检查是否存在
                os.makedirs(save_path)  # 创建多级目录
                print(f"目录 {save_path} 创建成功！")
            # else:
            #     print(f"目录 {save_path} 已存在，无需创建。")

            img_i = cv2.imread(file_i_path)
        
                # img_hued = Contrast(img_i,0-i)
                # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_hued.jpg"), img_hued)

            img_hueh = hue(img_i,7)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + "_hueh.jpg"), img_hueh)                        




def pixelate(image, pixel_size):
    """
    将图像像素化。

    :param image: 输入的BGR图像 (OpenCV格式)
    :param pixel_size: 像素块的大小
    :return: 像素化后的图像 (OpenCV格式)
    """
    # 获取图像尺寸
    height, width, _ = image.shape

    # 调整图像尺寸，使其可以被pixel_size整除
    new_width = width // pixel_size * pixel_size
    new_height = height // pixel_size * pixel_size
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 分块并计算平均颜色
    pixelated_image = resized_image.copy()
    for i in range(0, new_width, pixel_size):
        for j in range(0, new_height, pixel_size):
            # 获取当前像素块
            block = resized_image[j:j+pixel_size, i:i+pixel_size]
            average_color = block.mean(axis=0).mean(axis=0)  # 计算平均颜色

            # 填充像素块
            pixelated_image[j:j+pixel_size, i:i+pixel_size] = average_color

    # 将图像调整回原始尺寸
    pixelated_image = cv2.resize(pixelated_image, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def pixelate_image(rootpath, savepath, pixel_size=10):
    """
    遍历根目录下的所有图片，进行像素化处理并保存到目标目录。

    :param rootpath: 要处理的图片根目录
    :param savepath: 处理后图片的保存目录
    :param pixel_size: 像素块的大小
    """
    for root, _, files in os.walk(rootpath):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 读取图像
                img = cv2.imread(file_path)
                if img is None:
                    print(f"无法读取图像: {file_path}")
                    continue

                # 像素化处理
                pixelated_img = pixelate(img, pixel_size)

                # 构造保存路径
                relative_path = os.path.relpath(root, rootpath)
                save_dir = os.path.join(savepath, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                # 保存图像
                save_file = os.path.join(save_dir, f"pixelated_{file}")
                cv2.imwrite(save_file, pixelated_img)
                print(f"保存图像: {save_file}")

            except Exception as e:
                print(f"处理图像时出错: {file_path}, 错误信息: {e}") 

# def TestOnePic():
#     test_jpg_loc = r"data/A/firearms_001.jpg"
#     test_jpg = cv2.imread(test_jpg_loc)
#     cv2.imshow("Show Img", test_jpg)
#     cv2.waitKey(0)
#     img1 = Blur(test_jpg)
#     cv2.imshow("Img 1", img1)
#     cv2.waitKey(0)
#     img2 = GaussianNoise(test_jpg,0.4)
#     cv2.imshow("Img 2", img2)
#     cv2.waitKey(0)

def runs():
    root_path = r"dataset\O"

    save_path = r"dataset\96"
    
    YASUO(root_path,save_path,target_width=96, target_height=96)
    # Rotate_90_180_270(root_path,save_path)
    # D_dan_B(root_path,save_path)


    # save_path = r"dataset/Rotate"
    # Rotate_90_180_270(root_path,save_path)    #图像旋转---可任意角度

    # save_path = r"dataset/hue"
    # hue_image(root_path,save_path)    #图像色调扰动---可任意参数

    # save_path = r"dataset/D_B"
    # D_dan_B(root_path,save_path)    #图像明暗扰动---可任意参数

    # save_path = r"dataset/Cont"
    # Contrast_image(root_path,save_path)    #图像对比度扰动---可任意参数

    # save_path = r"dataset/hsv"
    # hsv_image(root_path,save_path)         #图像饱和度扰动---可任意参数

    # save_path = r"dataset/GS"
    # G_and_S(root_path,save_path)           #图像高斯和椒盐噪声扰动---可任意参数
    
    # pixelate_image(root_path,save_path,pixel_size=2)
if __name__ == "__main__":
    runs()
 

