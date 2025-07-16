import os
import sys
from datetime import datetime
from osgeo import ogr,gdal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QWidget, QFileDialog)
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QMessageBox

class ShapefileMerger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHP文件合并工具")
        self.setAcceptDrops(True)
        self.setGeometry(300, 300, 600, 400)
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 标题标签
        title_label = QLabel("拖放SHP文件到下方列表或点击添加按钮")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # 文件列表
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 2px dashed #aaa;
                min-height: 200px;
            }
        """)
        layout.addWidget(self.file_list)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("添加文件")
        self.add_button.clicked.connect(self.add_files)
        button_layout.addWidget(self.add_button)
        
        self.remove_button = QPushButton("移除选中")
        self.remove_button.clicked.connect(self.remove_files)
        button_layout.addWidget(self.remove_button)
        
        self.clear_button = QPushButton("清空列表")
        self.clear_button.clicked.connect(self.clear_files)
        button_layout.addWidget(self.clear_button)
        
        self.merge_button = QPushButton("合并文件")
        self.merge_button.clicked.connect(self.merge_files)
        self.merge_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.merge_button)
        
        layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        shp_files = []
        
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.shp'):
                shp_files.append(file_path)
        
        if shp_files:
            self.add_shp_files(shp_files)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择SHP文件", "", "Shapefiles (*.shp)"
        )
        if files:
            self.add_shp_files(files)
    
    def add_shp_files(self, files):
        existing_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        for file_path in files:
            if file_path not in existing_files:
                self.file_list.addItem(file_path)
        
        self.update_status(f"已添加 {len(files)} 个文件")
    
    def remove_files(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self.update_status("已移除选中文件")
    
    def clear_files(self):
        self.file_list.clear()
        self.update_status("已清空文件列表")
    
    def merge_files(self):
        if self.file_list.count() == 0:
            self.update_status("错误：请先添加文件", error=True)
            return
            # 弹出警告对话框
        reply = QMessageBox.question(
            self, '警告',
            "合并操作将修改原始文件，请确保已备份！\n是否继续合并？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            self.update_status("用户取消合并操作")
            return
        
        # 获取输出目录
        output_dir = os.path.dirname(self.file_list.item(0).text())
        # output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        
        # 获取文件列表
        shp_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        try:
            output_path = gdal_merge_shapefiles(shp_files, output_dir)
            self.update_status(f"合并成功！结果已保存到:\n{output_path}", success=True)
        except Exception as e:
            self.update_status(f"合并失败: {str(e)}", error=True)

    def update_status(self, message, error=False, success=False):
        if error:
            self.status_label.setStyleSheet("color: #F44336; font-size: 12px;")
        elif success:
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        else:
            self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        
        self.status_label.setText(message)


def delete_shapefile_gdal(shp_path):
    """
    使用GDAL驱动删除Shapefile数据集
    :param shp_path: .shp文件路径
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"文件不存在: {shp_path}")
    
    # GDAL删除会同时删除所有关联文件
    driver.DeleteDataSource(shp_path)
    print(f"已成功删除: {shp_path}及其关联文件")

def gdal_merge_shapefiles(shp_list, output_dir):
    """使用GDAL合并SHP文件的核心函数"""
    # 获取文件创建时间信息
    file_info = []
    for shp_path in shp_list:
        ctime = datetime.fromtimestamp(os.path.getctime(shp_path))
        file_info.append({
            'path': shp_path,
            'ctime': ctime,
            'name': os.path.splitext(os.path.basename(shp_path))[0]
        })
    
    # 按创建时间排序
    file_info.sort(key=lambda x: x['ctime'])
    earliest_name = file_info[0]['name']
    output_path = os.path.join(output_dir, f"{earliest_name}_merge.shp")
    
    # 初始化GDAL
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    # 获取第一个文件的字段定义
    first_ds = ogr.Open(file_info[0]['path'])
    first_layer = first_ds.GetLayer()
    layer_defn = first_layer.GetLayerDefn()
    
    # 创建输出文件
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer(
        'merged', 
        first_layer.GetSpatialRef(), 
        first_layer.GetGeomType()
    )
    
    # 复制字段定义
    for i in range(layer_defn.GetFieldCount()):
        out_layer.CreateField(layer_defn.GetFieldDefn(i))
    
    # 合并所有文件
    for info in file_info:
        ds = ogr.Open(info['path'])
        if ds is None:
            raise RuntimeError(f"无法打开文件: {info['path']}")
        
        layer = ds.GetLayer()
        
        # 验证字段结构
        if layer.GetLayerDefn().GetFieldCount() != layer_defn.GetFieldCount():
            raise ValueError(f"字段数量不匹配: {info['path']}")
        
        # 复制要素
        for feat in layer:
            out_feat = ogr.Feature(out_layer.GetLayerDefn())
            
            # 复制几何
            geom = feat.GetGeometryRef()
            if geom:
                out_feat.SetGeometry(geom.Clone())
            
            # 复制属性
            for i in range(layer_defn.GetFieldCount()):
                out_feat.SetField(i, feat.GetField(i))
            
            out_layer.CreateFeature(out_feat)
            out_feat = None
        
        ds = None
    
    # 清理资源
    out_ds = None
    first_ds = None
    for shp_path in shp_list:
        delete_shapefile_gdal(shp_path)
    return output_path

if __name__ == "__main__":
    gdal.UseExceptions()
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyleSheet("""
        QPushButton {
            padding: 5px 10px;
            min-width: 80px;
        }
        QListWidget::item {
            padding: 3px;
        }
    """)
    
    window = ShapefileMerger()
    window.show()
    sys.exit(app.exec_())