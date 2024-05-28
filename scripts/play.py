# 指定存放图片的文件夹路径
folder_path = '../result'

import cv2
import os
import time

# 获取文件夹下所有PNG图片的文件名
png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
png_files.sort()  # 按文件名排序

# 播放图片
for png_file in png_files:
    image_path = os.path.join(folder_path, png_file)
    image = cv2.imread(image_path)

    # set windows size
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(20)  # 每秒播放15张图片，即每张图片播放67毫秒

cv2.destroyAllWindows()