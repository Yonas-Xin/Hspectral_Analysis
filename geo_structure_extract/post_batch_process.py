import numpy as np
from post_processing1_edge_mask import mask_image
from skimage.morphology import remove_small_objects
from image_stretch import Gdal_Tool
from skimage.morphology import remove_small_holes
import cv2

def post_batch_process(img, top=0, bottom=0, left=0, right=0, small_hole_area=100000, small_obj_area=100000):
    mask_info = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    img = mask_image(img, mask_info=mask_info)
    img = img.astype(np.bool)
    img = remove_small_objects(img, small_obj_area)
    img = remove_small_holes(img, area_threshold=small_hole_area)
    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.erode(img, kernel, iterations=1)
    img = img.astype(np.bool)
    img = remove_small_objects(img, small_obj_area)
    img = img.astype(np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = img.astype(np.float32)
    return img


if __name__ == "__main__":
    input_path = r'c:\Users\85002\Desktop\构造解译示意\test\2222.tif'
    out_path = r'c:\Users\85002\Desktop\构造解译示意\test\12222.tif'
    gt = Gdal_Tool(input_path)
    img = gt.read_tif_to_image(1, to_int=False)
    top = 0
    bottom = 36
    left = 0
    right = 36
    img = post_batch_process(img, top, bottom, left, right)
    gt.save_tif(out_path, img)
