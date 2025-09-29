"""骨架化，提取二值线"""
import numpy as np
from skimage.morphology import skeletonize, label
from scipy import ndimage as ndi
from image_stretch import Gdal_Tool
# import matplotlib.pyplot as plt
def skeletonize_and_prune(binary_img, min_branch_length=30):
    """
    二值图骨架化并去掉短分支
    binary_img: 0/1 或 0/255 二值图
    min_branch_length: 最小保留分支长度（像素）
    """
    binary = (binary_img > 0)

    # Step 1: 骨架化
    skel = skeletonize(binary)

    # Step 2: 识别端点和分叉点
    struct = ndi.generate_binary_structure(2, 2)
    endpoints = []
    branchpoints = []

    for (y, x), v in np.ndenumerate(skel):
        if v:
            neighbors = np.sum(skel[y-1:y+2, x-1:x+2]) - 1
            if neighbors == 1:
                endpoints.append((y, x))
            elif neighbors > 2:
                branchpoints.append((y, x))

    # Step 3: 剪枝
    to_remove = set()
    for ep in endpoints:
        path = [ep]
        current = ep
        prev = None
        while True:
            # 找邻居
            neigh = [(ny, nx) for ny in range(current[0]-1, current[0]+2)
                                   for nx in range(current[1]-1, current[1]+2)
                                   if (ny, nx) != current and skel[ny, nx]]
            # 去掉回头像素
            neigh = [n for n in neigh if n != prev]
            if not neigh:
                break
            prev = current
            current = neigh[0]
            path.append(current)
            if current in branchpoints:
                break
        if len(path) < min_branch_length:
            to_remove.update(path)

    skel_clean = skel.copy()
    for p in to_remove:
        skel_clean[p] = False

    return skel_clean
def skeletonize_and_keep_main(binary_img, min_branch_length=30, keep_n_largest=1):
    """
    二值图骨架化并去掉短分支，只保留主体线
    binary_img: 0/1 或 0/255 二值图
    min_branch_length: 最小保留分支长度（像素）
    keep_n_largest: 保留的最大连通域条数
    """
    binary = (binary_img > 0)

    # Step 1: 骨架化
    skel = skeletonize(binary)

    # Step 2: 识别端点和分叉点
    endpoints = []
    branchpoints = []
    for (y, x), v in np.ndenumerate(skel):
        if v:
            neighbors = np.sum(skel[max(0, y-1):y+2, max(0, x-1):x+2]) - 1
            if neighbors == 1:
                endpoints.append((y, x))
            elif neighbors > 2:
                branchpoints.append((y, x))

    # Step 3: 剪枝
    to_remove = set()
    for ep in endpoints:
        path = [ep]
        current = ep
        prev = None
        while True:
            neigh = [(ny, nx) for ny in range(current[0]-1, current[0]+2)
                                 for nx in range(current[1]-1, current[1]+2)
                                 if (ny, nx) != current and 0 <= ny < skel.shape[0]
                                 and 0 <= nx < skel.shape[1] and skel[ny, nx]]
            neigh = [n for n in neigh if n != prev]
            if not neigh:
                break
            prev = current
            current = neigh[0]
            path.append(current)
            if current in branchpoints:
                break
        if len(path) < min_branch_length:
            to_remove.update(path)

    skel_clean = skel.copy()
    for p in to_remove:
        skel_clean[p] = False

    # Step 4: 连通域筛选（只保留主体线）
    labeled, num = label(skel_clean, return_num=True, connectivity=2)
    sizes = np.bincount(labeled.ravel())
    largest_labels = np.argsort(sizes)[::-1][1:keep_n_largest+1]  # 跳过背景 label=0

    skel_main = np.isin(labeled, largest_labels)

    return skel_main

input = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_dilate.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_xian.shp'
min_branch_length = 1000000

rgb = 1
if __name__ == "__main__":
    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=True)
    binary = skeletonize_and_prune(binary_img=img, min_branch_length=min_branch_length)
    # plt.imsave(out_path[:-4]+'.png', binary , cmap='gray')
    gt.skeleton_to_shp_from_raster(binary, out_path)