import numpy as np
import cv2

input = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_smallhole.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_erode.tif'
kernel_size = (5, 5)
iterations = 1

rgb = 1
if __name__ == "__main__":
    from image_stretch import Gdal_Tool
    import matplotlib.pyplot as plt
    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=False) 

    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img = cv2.erode(img, kernel, iterations=iterations) # 腐蚀
    img = img.astype(np.float32)
    plt.imsave(out_path[:-4]+'.png', img , cmap='gray')
    gt.save_tif(out_path, img)