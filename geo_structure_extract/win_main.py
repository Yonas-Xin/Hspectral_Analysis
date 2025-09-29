import sys, os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QAction ,QToolButton, QPushButton,
                             QLabel, QWidget, QMenu)
from PyQt5.QtGui import QPixmap, QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox
from dialog import SobelDialog, Wavelet2Dialog, CannyDialog, MaskImageDialog, RemoveSmallHoleDialog, RemoveSmallObjDialog, SkeletonizeDialog, ErodeDilateDialog
from functions import swt2_edge, canny_edge, sobel_edge, post_mask_image, post_remove_small_obj, post_erode_image, post_dilate_image, post_remove_small_hole, skeletonize_image

FUNCTIONS_DICT = {
    post_mask_image: MaskImageDialog,
    post_remove_small_obj: RemoveSmallObjDialog,
    post_remove_small_hole: RemoveSmallHoleDialog,
    post_erode_image: ErodeDilateDialog,
    post_dilate_image: ErodeDilateDialog,
    skeletonize_image: SkeletonizeDialog
}
class GeoStructure_Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("构造提取")
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
        title_label = QLabel("构造解译工具")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # 图片显示区域（替换原来的文件列表）
        self.image_label = QLabel("此处显示图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                min-height: 200px;
                background-color: #fafafa;
            }
        """)
        layout.addWidget(self.image_label)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.create_button_geostruct_areas(button_layout)
        self.create_button_postprocess_areas(button_layout)
        layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)

        self.clear_button = QPushButton("构造骨架提取")
        self.clear_button.clicked.connect(self.show_skeletonize_dialog)
        button_layout.addWidget(self.clear_button)
        
        main_widget.setLayout(layout)


    def create_button_geostruct_areas(self, layout):
        geostruct = QToolButton()
        geostruct.setText("      构造线提取      ")
        geostruct.setPopupMode(QToolButton.MenuButtonPopup)  # 主按钮 + 下拉菜单
        geostruct.setStyleSheet("QToolButton { font-size: 14px; padding: 6px 12px; }")

        data_menu = QMenu(geostruct)
        wavelet_action = QAction('二进小波', self)
        wavelet_action.triggered.connect(self.show_wavelet_dialog)
        data_menu.addAction(wavelet_action) 

        canny_action = QAction('Canny算子', self)
        canny_action.triggered.connect(self.show_canny_dialog)
        data_menu.addAction(canny_action)

        sobel_action = QAction('Sobel算子', self)
        sobel_action.triggered.connect(self.show_sobel_dialog)
        data_menu.addAction(sobel_action)
        geostruct.setMenu(data_menu)
        layout.addWidget(geostruct)

    def create_button_postprocess_areas(self, layout):
        geostruct = QToolButton()
        geostruct.setText("       结果后处理      ")
        geostruct.setPopupMode(QToolButton.MenuButtonPopup)  # 主按钮 + 下拉菜单
        geostruct.setStyleSheet("QToolButton { font-size: 14px; padding: 6px 12px; }")

        data_menu = QMenu(geostruct)

        import_action = QAction('二值边缘掩膜', self)
        import_action.triggered.connect(self.show_imagemask_dialog)
        data_menu.addAction(import_action)

        remove_obj_action = QAction('移除小物体', self)
        remove_obj_action.triggered.connect(self.show_removeobj_dialog)
        data_menu.addAction(remove_obj_action)

        remove_hole_action = QAction('移除小孔', self)
        remove_hole_action.triggered.connect(self.show_removehole_dialog)
        data_menu.addAction(remove_hole_action)

        erode_action = QAction('图像腐蚀', self)
        erode_action.triggered.connect(self.show_erode_dialog)
        data_menu.addAction(erode_action)

        dilate_action = QAction('图像膨胀', self)
        dilate_action.triggered.connect(self.show_dilate_dialog)
        data_menu.addAction(dilate_action)

        geostruct.setMenu(data_menu)
        layout.addWidget(geostruct)

    def show_wavelet_dialog(self):
        dialog = Wavelet2Dialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, binary = swt2_edge(**params)  # 解包参数并调用函数
            if binary is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(binary)  # 显示二值图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")
    
    def show_canny_dialog(self):
        dialog = CannyDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = canny_edge(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_sobel_dialog(self):
        dialog = SobelDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = sobel_edge(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_imagemask_dialog(self):
        dialog = MaskImageDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = post_mask_image(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_erode_dialog(self):
        dialog = ErodeDilateDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = post_erode_image(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_dilate_dialog(self):
        dialog = ErodeDilateDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = post_dilate_image(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_removehole_dialog(self):
        dialog = RemoveSmallHoleDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = post_remove_small_hole(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def show_removeobj_dialog(self):
        dialog = RemoveSmallObjDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = post_remove_small_obj(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")
    
    def show_skeletonize_dialog(self):
        dialog = SkeletonizeDialog(self)  # 复用CannyDialog, 可根据
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            result, edges = skeletonize_image(**params)  # 解包参数并调用函数
            if edges is not None:
                QMessageBox.information(self, "成功", f"处理完成，结果保存在: {result}")
                self.status_label.setText(f"处理完成: {result.split('/')[-1]}")
                self.arr2image(edges)  # 显示边缘图
            else:
                QMessageBox.critical(self, "错误", "处理失败，请检查输入参数。")
                self.status_label.setText("处理失败")

    def arr2image(self, data):
        data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8) # 默认将图像数据归一化到0-255
        h, w = data.shape
        qimage = QImage(data.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
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
    
    window = GeoStructure_Win()
    window.show()
    sys.exit(app.exec_())