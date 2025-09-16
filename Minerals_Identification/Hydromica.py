"""水云母化识别"""
"""水云母化通过2.2μm的Al-OH吸收峰与1.4μm辅助峰构建指数，利用逻辑运算和排他性规则"""
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import os
import numpy as np
import sys
from colors import rgb_colors

def water_mica_detection(input_tif, output_tif):
    try:
        # 1. 打开输入文件
        dataset = gdal.Open(input_tif)
        if dataset is None:
            raise ValueError("无法打开输入文件！")
        
        # 获取影像信息
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        bands = dataset.RasterCount
        geotrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        
        # 2. 读取所需波段（示例波段编号，需根据实际数据调整！）
        # 假设波段对应关系：
        # b146: 2.15μm | b147: 2.18μm | b149: 2.20μm 
        # b150: 2.21μm (Al-OH吸收中心) | b151: 2.22μm
        # b156: 2.30μm | b157: 2.32μm | b158: 2.35μm
        try:
            b146 = dataset.GetRasterBand(146).ReadAsArray().astype(float)
            b147 = dataset.GetRasterBand(147).ReadAsArray().astype(float)
            b149 = dataset.GetRasterBand(149).ReadAsArray().astype(float)
            b150 = dataset.GetRasterBand(150).ReadAsArray().astype(float)
            b151 = dataset.GetRasterBand(151).ReadAsArray().astype(float)
            b156 = dataset.GetRasterBand(156).ReadAsArray().astype(float)
            b157 = dataset.GetRasterBand(157).ReadAsArray().astype(float)
            b158 = dataset.GetRasterBand(158).ReadAsArray().astype(float)
        except Exception as e:
            raise IndexError(f"波段读取失败，请检查波段编号对应关系！错误：{str(e)}")
        
        condition = (
            ((b146 - b147) > 0) &
            ((b150 - b149) > 0) & 
            ((b151 - b150) > 0) & 
            ((b156 - b157) > 0) & 
            ((b158 - b157) > 0) 
        )
        print(f"成功识别像元总数：{np.sum(condition)}")

        # 4. 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_tif, cols, rows, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        out_dataset.SetGeoTransform(geotrans)
        out_dataset.SetProjection(proj)
        
        # 写入数据
        out_band = out_dataset.GetRasterBand(1)
        colors = gdal.ColorTable()
        colors.SetColorEntry(0, rgb_colors[0])  # 白色透明背景
        colors.SetColorEntry(1, rgb_colors[5]) # 紫色
        out_band.SetRasterColorTable(colors)
        out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        out_band.WriteArray(condition.astype(np.uint8))
        out_band.SetNoDataValue(0) # 设置0为nodata值
        
        # 关闭数据集
        dataset.FlushCache()
        dataset = None
        out_dataset = None
        print(f"处理完成！结果已保存至：{output_tif}")
        
    except Exception as e:
        print(f"处理失败！错误：{str(e)}", file=sys.stderr)
        sys.exit(1)
        
input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\haide_rp.dat"  # 替换为实际路径
output_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\predict_hydromica.tif"  # 替换为实际路径
if __name__ == "__main__":

    water_mica_detection(input_tif, output_tif)