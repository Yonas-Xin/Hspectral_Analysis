"""赤铁矿化识别"""
"""在使用envi格式的只有地理坐标数据时，不知道为什么无法正确读取坐标系，添加投影坐标后就可以正常读取了"""
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import os
import numpy as np
import sys
from colors import rgb_colors

def hematite_detection(input_tif, output_tif):
    try:
        dataset = gdal.Open(input_tif)
        if dataset is None:
            raise ValueError("无法打开输入文件！")
        
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        bands = dataset.RasterCount
        geotrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        try:
            b36 = dataset.GetRasterBand(36).ReadAsArray().astype(float)
            b20 = dataset.GetRasterBand(20).ReadAsArray().astype(float)
        except Exception as e:
            raise IndexError(f"波段读取失败，请检查波段编号对应关系！错误：{str(e)}")
        
        condition = (
            (b36 / (b20+1e-4)) > 1.8
        )
        print(f"成功识别像元总数：{np.sum(condition)}")

        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_tif, cols, rows, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        if geotrans is not None and proj is not None:
            out_dataset.SetGeoTransform(geotrans)
            out_dataset.SetProjection(proj)

        out_band = out_dataset.GetRasterBand(1)
        colors = gdal.ColorTable()
        colors.SetColorEntry(0, rgb_colors[0])  # 白色透明背景
        colors.SetColorEntry(1, rgb_colors[1]) # 红色
        out_band.SetRasterColorTable(colors)
        out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        out_band.WriteArray(condition.astype(np.uint8))
        out_band.SetNoDataValue(0) # 设置0为nodata值

        dataset.FlushCache()
        out_dataset.FlushCache()
        dataset = None
        out_dataset = None
        print(f"处理完成！结果已保存至：{output_tif}")
        
    except Exception as e:
        print(f"处理失败！错误：{str(e)}", file=sys.stderr)
        sys.exit(1)

        
input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\haide_rp.dat"  # 替换为实际路径
output_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\predict_hematite_1.8.tif"  # 替换为实际路径
if __name__ == "__main__":
    hematite_detection(input_tif, output_tif)