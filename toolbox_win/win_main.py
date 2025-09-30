import sys, os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QAction ,QToolButton, QPushButton,
                             QLabel, QWidget, QMenu)
from PyQt5.QtGui import QPixmap, QPixmap, QImage
from dialog import RandomCropDialog, run_algorithm, SuperpixelSamplingDialog, SmaccSamplingDialog, DdPredictionDialog,\
     ClusterFeaturesDialog, SampleOptimizeDialog, SampleCropDialog, SampleSplitDialog

class GeoStructure_Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TollBox for Hyperspectral Image Analysis")
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
        title_label1 = QLabel("对比学习样本裁剪")
        title_label1.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label1)
        
        # # 图片显示区域（替换原来的文件列表）
        # self.image_label = QLabel("Tollbox for Hyperspectral Image Analysis")
        # self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label.setStyleSheet("""
        #     QLabel {
        #         border: 2px dashed #aaa;
        #         min-height: 200px;
        #         background-color: #fafafa;
        #     }
        # """)
        # layout.addWidget(self.image_label)
        
        # 按钮区域
        self.button_rci = QPushButton("预训练采样")
        self.button_rci.clicked.connect(self.show_random_crop_dialog)
        layout.addWidget(self.button_rci)

        title_label2 = QLabel("半自动岩性样本集构建")
        title_label2.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label2)
        button_layout = QHBoxLayout()
        self.create_button_sampling_areas(button_layout)
        self.button_dd = QPushButton("特征降维")
        self.button_dd.clicked.connect(self.show_dd_prediction_dialog)
        button_layout.addWidget(self.button_dd)
        self.button_cf = QPushButton("样本聚类")
        self.button_cf.clicked.connect(self.show_cluster_features_dialog)
        button_layout.addWidget(self.button_cf)
        self.button_so = QPushButton("样本优化")
        self.button_so.clicked.connect(self.show_sample_optimize_dialog)
        button_layout.addWidget(self.button_so)
        layout.addLayout(button_layout)

        title_label3 = QLabel("监督学习样本集构建")
        title_label3.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label3)
        self.button_split = QPushButton("样本分割")
        self.button_split.clicked.connect(self.show_sample_split_dialog)
        layout.addWidget(self.button_split)

        self.button_clip = QPushButton("监督样本裁剪")
        self.button_clip.clicked.connect(self.show_sample_crop_dialog)
        layout.addWidget(self.button_clip)

            # ✅ 插入弹性空间
        layout.addStretch()
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)


    def create_button_sampling_areas(self, layout):
        geostruct = QToolButton()
        geostruct.setText("研究区自动采样")
        geostruct.setPopupMode(QToolButton.MenuButtonPopup)  # 主按钮 + 下拉菜单
        geostruct.setStyleSheet("QToolButton { font-size: 14px; padding: 6px 12px; }")

        data_menu = QMenu(geostruct)
        wavelet_action = QAction('超像素端元提取', self)
        wavelet_action.triggered.connect(self.show_ss_dialog)
        data_menu.addAction(wavelet_action) 

        canny_action = QAction('分块SMACC提取', self)
        canny_action.triggered.connect(self.show_smacc_dialog)
        data_menu.addAction(canny_action)

        geostruct.setMenu(data_menu)
        layout.addWidget(geostruct)

    def show_random_crop_dialog(self):
        dialog = RandomCropDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("random_crop", params)  # 解包参数并调用函数

    def show_ss_dialog(self):
        dialog = SuperpixelSamplingDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("superpixel_sampling", params)  # 解包参数并调用函数

    def show_smacc_dialog(self):
        dialog = SmaccSamplingDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("smacc_sampling", params)  # 解包参数并调用函数


    def show_dd_prediction_dialog(self):
        dialog = DdPredictionDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("dd_prediction", params)  # 解包参数并调用函数

    def show_cluster_features_dialog(self):
        dialog = ClusterFeaturesDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("cluster_features", params)  # 解包参数并调用函数

    def show_sample_optimize_dialog(self):
        dialog = SampleOptimizeDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("sample_optimize", params)  # 解包参数并调用函数

    def show_sample_crop_dialog(self):
        dialog = SampleCropDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("sample_crop", params)  # 解包参数并调用函数

    def show_sample_split_dialog(self):
        dialog = SampleSplitDialog(self)
        if dialog.exec_():  # 如果用户点击 OK
            params = dialog.get_params()
            run_algorithm("sample_split", params)  # 解包参数并调用函数

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