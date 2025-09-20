"""
图像显示和渲染工具模块
提供跨平台的图像显示功能，替代Windows的WM_PAINT消息处理
"""

import os
import sys
import warnings
from typing import Optional, Tuple, Any, Callable
import tempfile

# 尝试导入图像处理库
try:
    import numpy as np
except ImportError:
    # Mock numpy for development
    class MockNumPy:
        uint8 = 'uint8'
        uint16 = 'uint16'
        def array(self, data, dtype=None): return data
        def zeros(self, shape, dtype=None): return []
        def ascontiguousarray(self, arr): return arr
    np = MockNumPy()

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, some display features will be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, some image conversion features will be limited")

# 导入本地模块
try:
    from .gdal_utils import EnhancedRasterProcessor, quick_load_and_convert
    from .image_utils import RasterImageProcessor, quick_load_image
except ImportError:
    try:
        from gdal_utils import EnhancedRasterProcessor, quick_load_and_convert
        from image_utils import RasterImageProcessor, quick_load_image
    except ImportError:
        # 如果无法导入，提供mock实现
        class EnhancedRasterProcessor:
            def __init__(self): pass
            def load_image(self, *args): return False
            def cleanup_resources(self): pass
            def get_display_array(self): return None
        def quick_load_and_convert(*args): return None
        def quick_load_image(*args): return None


class ImageDisplayManager:
    """
    图像显示管理器
    提供跨平台的图像显示功能，相当于Windows的WM_PAINT消息处理
    """
    
    def __init__(self):
        self.current_image = None
        self.current_processor = None
        self.display_settings = {
            'auto_normalize': True,
            'convert_bgr_to_rgb': True,
            'background_color': (255, 255, 255),
            'fit_to_window': True
        }
        
    def __del__(self):
        """析构函数，清理资源"""
        self.cleanup()
        
    def cleanup(self):
        """清理所有资源"""
        if self.current_processor:
            self.current_processor.cleanup_resources()
            self.current_processor = None
        self.current_image = None
        
    def load_and_prepare_image(self, filepath: str, bands: Optional[Any] = None) -> bool:
        """
        加载并准备图像用于显示
        
        Args:
            filepath: 图像文件路径
            bands: 要显示的波段
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 清理之前的资源
            self.cleanup()
            
            # 创建处理器并加载图像
            self.current_processor = EnhancedRasterProcessor()
            
            if not self.current_processor.load_image(filepath, bands):
                return False
                
            # 获取显示用的图像数组
            self.current_image = self.current_processor.get_display_array(
                normalize=self.display_settings['auto_normalize']
            )
            
            return self.current_image is not None
            
        except Exception as e:
            warnings.warn(f"加载图像失败: {e}")
            self.cleanup()
            return False
            
    def paint_image(self, output_path: Optional[str] = None, 
                   figure_size: Tuple[int, int] = (10, 8),
                   title: Optional[str] = None) -> bool:
        """
        绘制图像 (相当于WM_PAINT消息处理)
        
        Args:
            output_path: 输出文件路径，None表示直接显示
            figure_size: 图像尺寸
            title: 图像标题
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.current_image is None:
            warnings.warn("没有加载的图像可以绘制")
            return False
            
        try:
            if MATPLOTLIB_AVAILABLE:
                return self._paint_with_matplotlib(output_path, figure_size, title)
            elif PIL_AVAILABLE:
                return self._paint_with_pil(output_path)
            else:
                # 保存为简单的文本表示
                return self._paint_as_text(output_path)
                
        except Exception as e:
            warnings.warn(f"绘制图像失败: {e}")
            return False
            
    def _paint_with_matplotlib(self, output_path: Optional[str], 
                              figure_size: Tuple[int, int], 
                              title: Optional[str]) -> bool:
        """使用matplotlib绘制图像"""
        try:
            plt.figure(figsize=figure_size)
            
            # 处理不同的图像格式
            if len(self.current_image.shape) == 2:
                # 灰度图像
                plt.imshow(self.current_image, cmap='gray')
            else:
                # 彩色图像
                plt.imshow(self.current_image)
                
            if title:
                plt.title(title)
            plt.axis('off')  # 不显示坐标轴
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"图像已保存到: {output_path}")
            else:
                plt.show()
                
            return True
            
        except Exception as e:
            warnings.warn(f"matplotlib绘制失败: {e}")
            return False
            
    def _paint_with_pil(self, output_path: Optional[str]) -> bool:
        """使用PIL绘制图像"""
        try:
            # 转换为PIL Image
            if hasattr(self.current_image, 'astype'):
                # numpy数组
                if len(self.current_image.shape) == 2:
                    # 灰度图
                    pil_image = Image.fromarray(self.current_image.astype(np.uint8), mode='L')
                elif len(self.current_image.shape) == 3:
                    if self.current_image.shape[2] == 3:
                        # RGB图像
                        pil_image = Image.fromarray(self.current_image.astype(np.uint8), mode='RGB')
                    elif self.current_image.shape[2] == 4:
                        # RGBA图像
                        pil_image = Image.fromarray(self.current_image.astype(np.uint8), mode='RGBA')
                    else:
                        # 其他多通道图像，取前3个通道
                        rgb_array = self.current_image[:, :, :3].astype(np.uint8)
                        pil_image = Image.fromarray(rgb_array, mode='RGB')
            else:
                # 非numpy数组，尝试直接创建
                pil_image = Image.fromarray(self.current_image)
                
            if output_path:
                pil_image.save(output_path)
                print(f"图像已保存到: {output_path}")
            else:
                # 在支持的系统上显示图像
                pil_image.show()
                
            return True
            
        except Exception as e:
            warnings.warn(f"PIL绘制失败: {e}")
            return False
            
    def _paint_as_text(self, output_path: Optional[str]) -> bool:
        """将图像信息保存为文本（最后的备选方案）"""
        try:
            info_text = f"图像信息:\n"
            if hasattr(self.current_image, 'shape'):
                info_text += f"形状: {self.current_image.shape}\n"
            if hasattr(self.current_image, 'dtype'):
                info_text += f"数据类型: {self.current_image.dtype}\n"
                
            # 添加一些统计信息
            try:
                if hasattr(self.current_image, 'min'):
                    info_text += f"最小值: {self.current_image.min()}\n"
                    info_text += f"最大值: {self.current_image.max()}\n"
                    info_text += f"平均值: {self.current_image.mean()}\n"
            except:
                pass
                
            if output_path:
                with open(output_path + '.txt', 'w', encoding='utf-8') as f:
                    f.write(info_text)
                print(f"图像信息已保存到: {output_path}.txt")
            else:
                print(info_text)
                
            return True
            
        except Exception as e:
            warnings.warn(f"文本输出失败: {e}")
            return False
            
    def get_image_info(self) -> dict:
        """获取当前图像信息"""
        if self.current_processor:
            return self.current_processor.get_info()
        return {}
        
    def update_display_settings(self, **kwargs):
        """更新显示设置"""
        self.display_settings.update(kwargs)
        
    def is_image_loaded(self) -> bool:
        """检查是否有图像已加载"""
        return self.current_image is not None


class SimpleImageViewer:
    """
    简单的图像查看器
    提供基本的图像显示功能
    """
    
    def __init__(self):
        self.display_manager = ImageDisplayManager()
        
    def view_image(self, filepath: str, bands: Optional[Any] = None, 
                  output_path: Optional[str] = None, title: Optional[str] = None) -> bool:
        """
        查看图像
        
        Args:
            filepath: 图像文件路径
            bands: 要显示的波段
            output_path: 输出路径，None表示直接显示
            title: 图像标题
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 加载图像
            if not self.display_manager.load_and_prepare_image(filepath, bands):
                print(f"无法加载图像: {filepath}")
                return False
                
            # 设置标题
            if title is None:
                title = os.path.basename(filepath)
                
            # 显示图像
            success = self.display_manager.paint_image(output_path, title=title)
            
            if success:
                # 输出图像信息
                info = self.display_manager.get_image_info()
                print(f"图像信息: {info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}, "
                      f"{info.get('bands', 'Unknown')} 波段")
                      
            return success
            
        except Exception as e:
            print(f"查看图像时出错: {e}")
            return False
            
    def cleanup(self):
        """清理资源"""
        if self.display_manager:
            self.display_manager.cleanup()


def quick_view_image(filepath: str, bands: Optional[Any] = None, 
                    output_path: Optional[str] = None, title: Optional[str] = None) -> bool:
    """
    快速查看图像的便捷函数
    
    Args:
        filepath: 图像文件路径
        bands: 要显示的波段
        output_path: 输出路径，None表示直接显示
        title: 图像标题
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    viewer = SimpleImageViewer()
    try:
        return viewer.view_image(filepath, bands, output_path, title)
    finally:
        viewer.cleanup()


def create_image_montage(filepaths: list, output_path: str, 
                        grid_size: Optional[Tuple[int, int]] = None,
                        titles: Optional[list] = None) -> bool:
    """
    创建图像蒙太奇（多图像拼接显示）
    
    Args:
        filepaths: 图像文件路径列表
        output_path: 输出文件路径
        grid_size: 网格大小 (rows, cols)，None表示自动计算
        titles: 图像标题列表
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    if not MATPLOTLIB_AVAILABLE:
        print("需要matplotlib来创建图像蒙太奇")
        return False
        
    try:
        num_images = len(filepaths)
        if num_images == 0:
            return False
            
        # 计算网格大小
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size
            
        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        # 加载和显示每个图像
        for i, filepath in enumerate(filepaths):
            if i >= len(axes):
                break
                
            try:
                # 快速加载图像
                image_array = quick_load_and_convert(filepath, normalize=True)
                if image_array is not None:
                    if len(image_array.shape) == 2:
                        axes[i].imshow(image_array, cmap='gray')
                    else:
                        axes[i].imshow(image_array)
                        
                    # 设置标题
                    if titles and i < len(titles):
                        axes[i].set_title(titles[i])
                    else:
                        axes[i].set_title(os.path.basename(filepath))
                else:
                    axes[i].text(0.5, 0.5, f'无法加载\n{os.path.basename(filepath)}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                               
            except Exception as e:
                axes[i].text(0.5, 0.5, f'加载错误\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[i].transAxes)
                           
            axes[i].axis('off')
            
        # 隐藏多余的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"图像蒙太奇已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"创建图像蒙太奇失败: {e}")
        return False


if __name__ == "__main__":
    # 测试代码
    print("Display utilities module loaded successfully")
    
    # 测试显示管理器
    display_manager = ImageDisplayManager()
    print(f"Display manager created: {not display_manager.is_image_loaded()}")
    
    # 测试查看器
    viewer = SimpleImageViewer()
    print("Simple viewer created successfully")
    
    # 清理测试
    display_manager.cleanup()
    viewer.cleanup()
    print("Cleanup test completed")