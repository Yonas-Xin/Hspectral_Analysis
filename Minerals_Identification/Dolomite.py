"""碳酸盐化识别-白云石"""
"""针对碳酸盐化依托2.35μm的CO32-离子强吸收峰及2.55μm伴随峰设计波段组合构建指数，碳酸盐化在本区包括白云石化与方解石化，特征相似但主吸收峰位置有所差异"""
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import os
import numpy as np
import sys
from colors import rgb_colors

def mineral_classification(input_tif, output_tif):
    """同步计算方解石和白云石分布，并输出分类结果"""
    try:
        # 打开输入文件
        dataset = gdal.Open(input_tif)
        if not dataset:
            raise ValueError("输入文件无法打开！")

        cols = dataset.RasterXSize
        rows = dataset.RasterYSize

        # ===================== 白云石识别 =====================
        b125 = dataset.GetRasterBand(125).ReadAsArray().astype(float)
        b135 = dataset.GetRasterBand(135).ReadAsArray().astype(float)
        b137 = dataset.GetRasterBand(137).ReadAsArray().astype(float)
        b138 = dataset.GetRasterBand(138).ReadAsArray().astype(float)
        b144 = dataset.GetRasterBand(144).ReadAsArray().astype(float)
        b145 = dataset.GetRasterBand(145).ReadAsArray().astype(float)
        b147 = dataset.GetRasterBand(147).ReadAsArray().astype(float)
        b148 = dataset.GetRasterBand(148).ReadAsArray().astype(float)
        b152 = dataset.GetRasterBand(152).ReadAsArray().astype(float)
        b153 = dataset.GetRasterBand(153).ReadAsArray().astype(float)
        b154 = dataset.GetRasterBand(154).ReadAsArray().astype(float)
        b155 = dataset.GetRasterBand(155).ReadAsArray().astype(float)
        b156 = dataset.GetRasterBand(156).ReadAsArray().astype(float)
        b157 = dataset.GetRasterBand(157).ReadAsArray().astype(float)
        b158 = dataset.GetRasterBand(158).ReadAsArray().astype(float)
        b161 = dataset.GetRasterBand(161).ReadAsArray().astype(float)

        dolomite_mask = (
            ((b125 - b135) > 0) & 
            ((b137 - b135) > 0) & 
            ((b147 - b155) > 0) & 
            ((b158 - b155) > 0) & 
            ((b158 - b161) > 0) & 
            ((b145 - b144) > 0) & 
            ((b152 - b153) > 0) & 
            ((b153 - b154) > 0) & 
            ((b156 - b155) > 0) & 
            ((b138 - b144) > 0) & 
            ((b157 - b158) > 0) & 
            ((b147 - b148) > 0)
        )
        print(f"成功识别像元总数：{np.sum(dolomite_mask)}")
        # ===================== 输出结果 =====================
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_tif,
            cols,
            rows,
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        out_ds.SetGeoTransform(dataset.GetGeoTransform())
        out_ds.SetProjection(dataset.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        colors = gdal.ColorTable()
        colors.SetColorEntry(0, rgb_colors[0])      # 背景-透明
        colors.SetColorEntry(1, rgb_colors[3])  # 白云石-蓝色
        out_band.SetRasterColorTable(colors)
        out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        out_band.WriteArray(dolomite_mask)
        out_band.SetNoDataValue(0)
        
        out_ds = None
        dataset = None
        print(f"处理完成！结果保存至：{output_tif}")

    except Exception as e:
        print(f"处理失败：{str(e)}", file=sys.stderr)
        sys.exit(1)
        
input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\haide\haide_rp.dat"   # 替换为实际路径
output_tif = r"c:\Users\85002\OneDrive - cugb.edu.cn\Word\小组项目\240922张川-铀矿探测\zy_kuangwu\predict_dolomite.tif"  # 替换为实际路径
if __name__ == "__main__":
    mineral_classification(input_tif, output_tif)