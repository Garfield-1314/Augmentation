import cv2
import numpy as np
# 读取图像
image = cv2.imread('test.jpg')

# 亮度扰动
brightness_factor = 0.8  # 亮度增强系数
brightened_image = cv2.multiply(image, brightness_factor)

# 对比度扰动
contrast_factor = 0.7 # 对比度增强系数
contrasted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# 饱和度扰动
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
saturation_factor =0.7  # 饱和度增强系数
hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)
saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# 色调扰动
hue_shift = 200  # 色调偏移量
hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue_image[:, :, 0] = (hue_image[:, :, 0] + hue_shift) % 180
hue_shifted_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
# 显示结果
# cv2.imshow('Original', image)
# cv2.imshow('Brightened', brightened_image)
# cv2.imshow('Contrasted', contrasted_image)
# cv2.imshow('Saturated', saturated_image)
cv2.imshow('Hue Shifted', hue_shifted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()