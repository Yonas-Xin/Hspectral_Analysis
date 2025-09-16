"""绿泥石化识别"""
"""针对绿泥石化则利用2.3μm附近的Mg-OH特征吸收峰及2.275μm伴随峰构建指数，利用逻辑运算和排他性规则"""
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import os
import numpy as np
import sys
from colors import rgb_colors

def chlorite_detection(input_tif, output_tif):
    try:
        # 打开输入文件
        dataset = gdal.Open(input_tif)
        if dataset is None:
            raise ValueError("无法打开输入文件！")
        
        # 获取影像信息
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        geotrans = dataset.GetGeoTransform()    
        proj = dataset.GetProjection()
        
        # 读取波段数据
        try:
            b84 = dataset.GetRasterBand(84).ReadAsArray().astype(float)
            b123 = dataset.GetRasterBand(123).ReadAsArray().astype(float)
            b146 = dataset.GetRasterBand(146).ReadAsArray().astype(float)
            b151 = dataset.GetRasterBand(151).ReadAsArray().astype(float)
            b152 = dataset.GetRasterBand(152).ReadAsArray().astype(float)
            b157 = dataset.GetRasterBand(157).ReadAsArray().astype(float)
            b162 = dataset.GetRasterBand(162).ReadAsArray().astype(float)
        except Exception as e:
            raise IndexError(f"波段读取失败，请检查波段编号对应关系！错误：{str(e)}")
        
        # 计算条件
        condition = (
            ((b123 - b84) > 0) &
            ((b146 - b151) > 0) &
            ((b152 - b157) > 0) &
            ((b162 - b157) > 0) & 
            ((b152 - b151) > 0) 
        )
        print(f"成功识别像元总数：{np.sum(condition)}")
        
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_tif, 
            cols, 
            rows, 
            1, 
            gdal.GDT_Byte,
            options=['COMPRESS=LZW']
        )
        out_dataset.SetGeoTransform(geotrans)
        out_dataset.SetProjection(proj)
        

        # 写入数据
        out_band = out_dataset.GetRasterBand(1)
        colors = gdal.ColorTable()
        colors.SetColorEntry(0, rgb_colors[0])  # 白色透明背景
        colors.SetColorEntry(1, rgb_colors[4]) # 绿色
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
output_tif = r"c:\Users\85002\OneDrive - cugb.edu.cn\Word\小组项目\240922张川-铀矿探测\zy_kuangwu\predict_chlorite.tif"  # 替换为实际路径
if __name__ == "__main__":
    chlorite_detection(input_tif, output_tif)