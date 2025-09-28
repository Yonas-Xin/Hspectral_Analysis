import cv2
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

input_tif = r'C:\Users\85002\Desktop\TempDIR\out2.dat'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_sobel_4.tif'
th = 10 # 阈值分割，0-255，用于将边缘结果二值化的阈值

DOWN_SAMPLE_FUNC = "NEAREST" # "LINEAR", "CUBIC", "NEAREST"
DOWN_SAMPLE_FACTOR = 1 # 降采样倍数
stretch = "Linear" # Linear_2% or Linear
rgb = (1,2,3) # rgb 组合，从1开始。(1,2,3) or 1
if '__main__' == __name__:
    gt = Gdal_Tool(input_tif)
    img = gt.read_tif_to_image(rgb, stretch=stretch, to_int=True, to_gray=True)
    if DOWN_SAMPLE_FACTOR > 1: # 降采样
        img = gt.down_sample(img, factor=DOWN_SAMPLE_FACTOR, FUNC=DOWN_SAMPLE_FUNC)

    # 计算Sobel卷积结果
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    Scale_absX = cv2.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 图像混合, unit8格式
    result[result > th] = 255

    plt.imsave(out_path[:-4]+'.png', result , cmap='gray')
    gt.save_tif(out_path, result, factor=DOWN_SAMPLE_FACTOR if DOWN_SAMPLE_FACTOR > 1 else None)