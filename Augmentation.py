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
def compress_img_CV(img, compress_rate=0.5, show=False):
        heigh, width = img.shape[:2]
        # 双三次插值
        img_resize = cv2.resize(img, (int(heigh*compress_rate), int(width*compress_rate)),
                                interpolation=cv2.INTER_AREA)
        return img_resize

def YASUO_80(rootpath,savepath):
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

            img_yasuo = compress_img_CV(img_i,compress_rate=0.5)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_80.jpg"),img_yasuo)
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
            i=0.1
            # while(i<=0.2):
            #     img_dar = Darker_Brighter(img_i,1-i)
            #     cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_dar.jpg"), img_dar)
            #     i+=0.1

            i=0.1
            while(i<=0.7):             
                img_bar = Darker_Brighter(img_i,1+i)
                cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_bar.jpg"), img_bar)                        
                i+=0.1
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
            
            i=0.1
            while(i<=0.3):
                img__Contrastd = Contrast(img_i,1-i)
                cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_Contrastd.jpg"), img__Contrastd)

                img__Contrasth = Contrast(img_i,1+i)
                cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_Contrasth.jpg"), img__Contrasth)                        
                i+=0.1
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
            
            i=0.1
            while(i<=0.3):
                img_hsvd = hsv(img_i,1-i)
                cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_hsvd.jpg"), img_hsvd)

                img_hsvh = hsv(img_i,1+i)
                cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) + "_hsvh.jpg"), img_hsvh)                        
                i+=0.1
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

if __name__ == "__main__":

    root_path = r"dataset/origin"

    save_path = r"dataset/YASUO_80"
    YASUO_80(root_path,save_path)   #图像压缩---任意比例

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
    
 

