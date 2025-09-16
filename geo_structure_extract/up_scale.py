import cv2
import numpy as np
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

def downsample(img, factor=2):
    """
    将图像的尺度变大，并保持大小不变
    输入:
        img: numpy.ndarray, float32, (H,W) 或 (H,W,3)，范围 [0,1]
        factor: int, >=2
    输出:
        numpy.ndarray, float32, (H,W) 或 (H,W,3)，与原图大小相同
    """
    H, W = img.shape[:2]
    # 先缩小
    small = cv2.resize(img, (W // factor, H // factor), interpolation=cv2.INTER_AREA)
    # 再放大回原始大小
    # blurred = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return small.astype(np.float32)

input = r'C:\Users\85002\Desktop\TempDIR\out.dat'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\upscale.tif'
factor = 5
rgb = (1,2,3)
if __name__ == "__main__":

    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=False, to_gray=False)
    img = downsample(img, factor=factor)
    plt.imsave(out_path[0:-4]+'.png', img)
    img = img.transpose(2,0,1)
    gt.save_tif(out_path, img, factor=5)