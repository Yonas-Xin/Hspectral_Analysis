
import numpy as np
import cv2
from skimage.morphology import remove_small_holes
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

input = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_smallobj.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_smallhole.tif'
area = 1000000

rgb = 1
if __name__ == "__main__":
    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=False)

    img = img.astype(np.bool)
    img = remove_small_holes(img, area_threshold=area) # 填充
    img = img.astype(np.float32)
    plt.imsave(out_path[:-4]+'.png', img , cmap='gray')
    gt.save_tif(out_path, img)