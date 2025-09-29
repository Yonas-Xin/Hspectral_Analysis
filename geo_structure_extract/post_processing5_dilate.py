import numpy as np
import cv2
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

input = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_erode.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_dilate.tif'
kernel_size = (5, 5)
iterations = 1

rgb = 1
if __name__ == "__main__":

    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=False)

    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = img.astype(np.float32)
    plt.imsave(out_path[:-4]+'.png', img , cmap='gray')
    gt.save_tif(out_path, img)