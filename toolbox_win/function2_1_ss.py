import sys, os
sys.path.append('.')
from core import Hyperspectral_Image
import argparse
def superpixel_sampling(input_tif, out_shp, max_samples=30, n_segments=512, enhance_func='MNF', embedding_nums=12,
                           compactness=25, ppi_niters=2000, ppi_threshold=0, ppi_centered=False):
    '''
    超像素分割+随机采样
    :param input_tif: 输入影像路径
    :param out_shp: 输出shp路径
    :param max_samples: 控制每个超像素最大采样量
    :param n_segments: 调整超像素数量
    :param enhance_func: 可选'MNF' "PCA" 控制ppi计算时的降维方法
    :param embedding_nums: 控制降维维度
    :param compactness: 超像素分割的参数，调整超像素的紧密度
    :param ppi_niters: ppi迭代次数
    :param ppi_threshold: ppi阈值
    :param ppi_centered: ppi是否中心化
    :return:
    '''
    try:
        img = Hyperspectral_Image()
        img.init(input_tif, init_fig=True)  # 使用原始数据的增强影像
        print(f'The number of pixels: {img.rows * img.cols}')
        slic_label, slic_img = img.slic(n_segments=n_segments, compactness=compactness, n_components=embedding_nums)

        if enhance_func == 'MNF':
            img.image_enhance(f=enhance_func, n_components=embedding_nums)
        res = img.superpixel_sampling(slic_label, img.enhance_data, max_samples=max_samples,
                                    niters=ppi_niters, threshold=ppi_threshold, centered=ppi_centered)
        img.create_vector(res, out_shp) # 创建单个shp文件，二维矩阵转点shp文件
        return True, "采样成功"
    except Exception as e:
        return False, f"采样失败: {e}"
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", type=str, required=True)
    parser.add_argument("--out_shp", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--n_segments", type=int, default=512)
    parser.add_argument("--enhance_func", type=str, default='MNF')
    parser.add_argument("--embedding_nums", type=int, default=12)
    parser.add_argument("--compactness", type=int, default=25)
    parser.add_argument("--ppi_niters", type=int, default=2000)
    parser.add_argument("--ppi_threshold", type=int, default=0)
    parser.add_argument("--ppi_centered", type=bool, default=False)
    args = parser.parse_args()

    superpixel_sampling(
        input_tif=os.path.abspath(args.input_tif),
        out_shp=os.path.abspath(args.out_shp),
        max_samples=args.max_samples,
        n_segments=args.n_segments,
        enhance_func=args.enhance_func,
        embedding_nums=args.embedding_nums,
        compactness=args.compactness,
        ppi_niters=args.ppi_niters,
        ppi_threshold=args.ppi_threshold,
        ppi_centered=args.ppi_centered
    )