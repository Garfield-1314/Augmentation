import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import random




def Cover(rootpath,back_path):
    num=0
    for ab,bb,cb in os.walk(back_path):
        for file_b_b in cb:
            file_b_path = os.path.join(ab,file_b_b)
            print(file_b_path)    
            save_loc = rootpath
            for a,b,c in os.walk(rootpath):
                for file_i in c:
                    file_i_path = os.path.join(a,file_i)
                    print(file_i_path)
                    split = os.path.split(file_i_path)
                    dir_loc = os.path.split(split[0])[1]
                    save_path = os.path.join(save_loc,dir_loc)
                    
                    img1 = cv2.imread(file_b_path)  

                    img2 = cv2.imread(file_i_path)

                    rows, cols, channels = img2.shape  # 获取图像2的属性
                    roi = img1[0:rows, 0:cols]  # 选择roi范围

                    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                    ret, mask = cv2.threshold(img2gray, 255, 255, cv2.THRESH_BINARY)  # 设置阈值，大于175的置为255，小于175的置为0
                    mask_inv = cv2.bitwise_not(mask)  # 非运算，mask取反

                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)  #删除了ROI中的logo区域
                    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv) #删除了logo中的空白区域

                    dst = cv2.add(img1_bg, img2_fg)
                    x=random.randint(1, 47)
                    y=random.randint(1, 47)
                    # x=30
                    # y=30         
                    img1[0+x:rows+x, 0+y:cols+y] = dst
                    # cv2.imshow('res', img1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    num= num+1
                    cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(num) + "_cover.jpg"), img1)

                    # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) +  "_bar.jpg"), img_bar)                        

if __name__ == "__main__":
    back_path = r"P"
    root_path = r"abc_80"

    Cover(root_path,back_path)
