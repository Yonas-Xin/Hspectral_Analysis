import cv2
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

input_tif = r'c:\Users\85002\Desktop\TempDIR\test2\upscale.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use.tif'
threshold1 = 50
threshold2 = 100

stretch = "Linear_2%" # Linear_2% or Linear
rgb = (1,2,3) # rgb 组合，从1开始。(1,2,3) or 1
if '__main__' == __name__:
    gt = Gdal_Tool(input_tif)
    img = gt.read_tif_to_image(rgb, stretch=stretch, to_int=True, to_gray=False)
    edge = cv2.Canny(img,threshold1,threshold2) # 第一个阈值用于连接断线，第二个阈值用于判断明显的边缘
    plt.imsave(out_path[:-4]+'.png', img , cmap='gray')
    gt.save_tif(out_path, img)
