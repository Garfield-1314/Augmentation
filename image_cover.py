import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import random




def Cover(rootpath,back_path,savepath):
    num=0
    for ab,bb,cb in os.walk(back_path):
        for file_b_b in cb:
            file_b_path = os.path.join(ab,file_b_b)
            print(file_b_path)    
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

                    img1 = cv2.imread(file_b_path)  

                    img2 = cv2.imread(file_i_path)

                    rowsb, colsb ,channelsb = img1.shape

                    rows, cols, channels = img2.shape  # 获取图像2的属性
                    
                    print(rowsb,colsb,rows,cols)

                    startx=random.randint(5, rowsb-rows-5)
                    starty=random.randint(5, colsb-cols-5)
                    
                    print(startx,starty)
                    
                    roi = img1[startx:startx+rows, starty:starty+cols]  # 选择roi范围

                    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                    ret, mask = cv2.threshold(img2gray, 255, 255, cv2.THRESH_BINARY)  # 设置阈值，大于175的置为255，小于175的置为0
                    mask_inv = cv2.bitwise_not(mask)  # 非运算，mask取反

                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)  #删除了ROI中的logo区域
                    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv) #删除了logo中的空白区域

                    dst = cv2.add(img1_bg, img2_fg)    
                    img1[0+startx:rows+startx, 0+starty:cols+starty] = dst
                    # cv2.imshow('res', img1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    num= num+1
                    cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(num) + "_cover.jpg"), img1)

                    # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_" + str(i) +  "_bar.jpg"), img_bar)                        

if __name__ == "__main__":  #背景图片必须大于目标图片
    back_path = r"dataset/background"
    root_path = r"dataset/YASUO_80"
    save_path = r"dataset/used"
    Cover(root_path,back_path,save_path)
