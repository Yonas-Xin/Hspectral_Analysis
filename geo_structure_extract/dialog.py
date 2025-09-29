from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog,
                             QSpinBox, QComboBox, QDialogButtonBox,
                             QCheckBox)
from PyQt5.QtCore import Qt

class BaseDialog(QDialog):
    """基础对话框类"""
    def __init__(self, parent=None, title="参数设置"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 300)
        
    def create_button_layout(self, layout):
        """创建确定和取消按钮布局"""
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def layout_choose_file(self, layout):
        """创建选择文件的布局"""
        file_layout = QHBoxLayout()
        self.input_tif = QLineEdit()
        self.input_tif.setPlaceholderText("请选择输入文件...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(QLabel("输入文件:"))
        file_layout.addWidget(self.input_tif)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)
        return file_layout
    
    def layout_choose_out_file(self, layout):
        """创建选择输出文件的布局"""
        out_layout = QHBoxLayout()
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("输出文件路径...")
        browse_out_btn = QPushButton("浏览")
        browse_out_btn.clicked.connect(self.browse_out_file)
        out_layout.addWidget(QLabel("输出文件:"))
        out_layout.addWidget(self.out_path)
        out_layout.addWidget(browse_out_btn)
        layout.addLayout(out_layout)
        return out_layout
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "Data Files (*.tif *.dat)")
        if file_path:
            self.input_tif.setText(file_path)

    def browse_out_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出文件", "", "TIFF Files (*.tif)")
        if file_path:
            self.out_path.setText(file_path)
    

class Wavelet2Dialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "构造提取")
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()

        # 文件路径选择
        self.layout_choose_file(layout)
        # 输出路径
        self.layout_choose_out_file(layout)
        # level
        level_layout = QHBoxLayout()
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 20)
        self.level_spin.setValue(5)  # 默认值
        level_layout.addWidget(QLabel("level:"))
        level_layout.addWidget(self.level_spin)
        layout.addLayout(level_layout)

        # use_modulus_maxima
        self.use_modulus_check = QCheckBox("启用模极大值检测")
        self.use_modulus_check.setChecked(True)  # 默认值
        layout.addWidget(self.use_modulus_check)

        # wavelet
        wavelet_layout = QHBoxLayout()
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(['haar', 'db1', 'db2', 'sym2', 'coif1', 'bior2.2'])
        self.wavelet_combo.setCurrentText('haar')
        wavelet_layout.addWidget(QLabel("wavelet:"))
        wavelet_layout.addWidget(self.wavelet_combo)
        layout.addLayout(wavelet_layout)

        # stretch
        stretch_layout = QHBoxLayout()
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(['Linear_2%', 'Linear'])
        self.stretch_combo.setCurrentText('Linear_2%')
        stretch_layout.addWidget(QLabel("stretch:"))
        stretch_layout.addWidget(self.stretch_combo)
        layout.addLayout(stretch_layout)

        # rgb
        rgb_layout = QHBoxLayout()
        self.rgb_edit = QLineEdit("1,2,3")  # 默认值
        rgb_layout.addWidget(QLabel("RGB通道:"))
        rgb_layout.addWidget(self.rgb_edit)
        layout.addLayout(rgb_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        self.setLayout(layout)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "level": self.level_spin.value(),
            "use_modulus_maxima": self.use_modulus_check.isChecked(),
            "wavelet": self.wavelet_combo.currentText(),
            "stretch": self.stretch_combo.currentText(),
            "rgb": tuple(map(int, self.rgb_edit.text().split(",")))
        }
    
class CannyDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canny 边缘提取参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 文件路径选择
        self.layout_choose_file(layout)
        # 输出路径
        self.layout_choose_out_file(layout)

        # threshold1
        t1_layout = QHBoxLayout()
        self.threshold1_spin = QSpinBox()
        self.threshold1_spin.setRange(0, 255)
        self.threshold1_spin.setValue(50)
        t1_layout.addWidget(QLabel("Threshold1:"))
        t1_layout.addWidget(self.threshold1_spin)
        layout.addLayout(t1_layout)

        # threshold2
        t2_layout = QHBoxLayout()
        self.threshold2_spin = QSpinBox()
        self.threshold2_spin.setRange(0, 255)
        self.threshold2_spin.setValue(100)
        t2_layout.addWidget(QLabel("Threshold2:"))
        t2_layout.addWidget(self.threshold2_spin)
        layout.addLayout(t2_layout)

        # DOWN_SAMPLE_FUNC
        dsf_layout = QHBoxLayout()
        self.downsample_func_combo = QComboBox()
        self.downsample_func_combo.addItems(["LINEAR", "CUBIC", "NEAREST"])
        self.downsample_func_combo.setCurrentText("NEAREST")
        dsf_layout.addWidget(QLabel("降采样方法:"))
        dsf_layout.addWidget(self.downsample_func_combo)
        layout.addLayout(dsf_layout)

        # DOWN_SAMPLE_FACTOR
        dsf_factor_layout = QHBoxLayout()
        self.downsample_factor_spin = QSpinBox()
        self.downsample_factor_spin.setRange(1, 20)
        self.downsample_factor_spin.setValue(5)
        dsf_factor_layout.addWidget(QLabel("降采样倍数:"))
        dsf_factor_layout.addWidget(self.downsample_factor_spin)
        layout.addLayout(dsf_factor_layout)

        # stretch
        stretch_layout = QHBoxLayout()
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(["Linear_2%", "Linear"])
        self.stretch_combo.setCurrentText("Linear")
        stretch_layout.addWidget(QLabel("图像拉伸:"))
        stretch_layout.addWidget(self.stretch_combo)
        layout.addLayout(stretch_layout)

        # rgb
        rgb_layout = QHBoxLayout()
        self.rgb_edit = QLineEdit("1,2,3")
        rgb_layout.addWidget(QLabel("RGB通道:"))
        rgb_layout.addWidget(self.rgb_edit)
        layout.addLayout(rgb_layout)

        # 确认/取消
        self.create_button_layout(layout)

        self.setLayout(layout)

    # ===== 获取参数 =====
    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "th1": self.threshold1_spin.value(),
            "th2": self.threshold2_spin.value(),
            "down_sample_factor": self.downsample_factor_spin.value(),
            "down_sample_func": self.downsample_func_combo.currentText(),
            "stretch": self.stretch_combo.currentText(),
            "rgb": tuple(map(int, self.rgb_edit.text().split(",")))
        }

class SobelDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sobel 边缘提取参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 文件路径选择
        self.layout_choose_file(layout)
        # 输出路径
        self.layout_choose_out_file(layout)

        # 阈值 th
        th_layout = QHBoxLayout()
        self.th_spin = QSpinBox()
        self.th_spin.setRange(0, 255)
        self.th_spin.setValue(10)  # 默认值
        th_layout.addWidget(QLabel("阈值 th:"))
        th_layout.addWidget(self.th_spin)
        layout.addLayout(th_layout)

        # 降采样倍数
        dsf_layout = QHBoxLayout()
        self.down_sample_factor_spin = QSpinBox()
        self.down_sample_factor_spin.setRange(1, 20)
        self.down_sample_factor_spin.setValue(1)
        dsf_layout.addWidget(QLabel("降采样倍数:"))
        dsf_layout.addWidget(self.down_sample_factor_spin)
        layout.addLayout(dsf_layout)

        # 降采样方法
        dsf_func_layout = QHBoxLayout()
        self.down_sample_func_combo = QComboBox()
        self.down_sample_func_combo.addItems(["LINEAR", "CUBIC", "NEAREST"])
        self.down_sample_func_combo.setCurrentText("NEAREST")
        dsf_func_layout.addWidget(QLabel("降采样方法:"))
        dsf_func_layout.addWidget(self.down_sample_func_combo)
        layout.addLayout(dsf_func_layout)

        # stretch
        stretch_layout = QHBoxLayout()
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(["Linear_2%", "Linear"])
        self.stretch_combo.setCurrentText("Linear")
        stretch_layout.addWidget(QLabel("图像拉伸:"))
        stretch_layout.addWidget(self.stretch_combo)
        layout.addLayout(stretch_layout)

        # rgb
        rgb_layout = QHBoxLayout()
        self.rgb_edit = QLineEdit("1,2,3")
        rgb_layout.addWidget(QLabel("RGB通道:"))
        rgb_layout.addWidget(self.rgb_edit)
        layout.addLayout(rgb_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        self.setLayout(layout)

    # ===== 获取参数 =====
    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "th": self.th_spin.value(),
            "down_sample_factor": self.down_sample_factor_spin.value(),
            "down_sample_func": self.down_sample_func_combo.currentText(),
            "stretch": self.stretch_combo.currentText(),
            "rgb": tuple(map(int, self.rgb_edit.text().split(",")))
        }
    
class MaskImageDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("边缘掩膜参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 输入文件
        self.layout_choose_file(layout)

        # 输出文件
        self.layout_choose_out_file(layout)

        # 四个边缘参数
        for name, default in zip(["Top", "Bottom", "Left", "Right"], [0,0,0,0]):
            hlayout = QHBoxLayout()
            spin = QSpinBox()
            spin.setRange(0, 10000)
            spin.setValue(default)
            setattr(self, f"{name.lower()}_spin", spin)
            hlayout.addWidget(QLabel(f"{name}边缘宽度:"))
            hlayout.addWidget(spin)
            layout.addLayout(hlayout)

        # 确认/取消
        self.create_button_layout(layout)

        self.setLayout(layout)

    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "top": self.top_spin.value(),
            "bottom": self.bottom_spin.value(),
            "left": self.left_spin.value(),
            "right": self.right_spin.value()
        }
    
class RemoveSmallObjDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("去除小物体参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        # 输入文件
        self.layout_choose_file(layout)

        # 输出文件
        self.layout_choose_out_file(layout)

        # 面积阈值
        area_layout = QHBoxLayout()
        self.area_spin = QSpinBox()
        self.area_spin.setRange(0, 100000000)
        self.area_spin.setValue(1000000)
        area_layout.addWidget(QLabel("小物体面积阈值:"))
        area_layout.addWidget(self.area_spin)
        layout.addLayout(area_layout)

        # 确认/取消
        self.create_button_layout(layout)
        self.setLayout(layout)

    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "area": self.area_spin.value()
        }

class RemoveSmallHoleDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("填充小孔洞参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        # 输入文件
        self.layout_choose_file(layout)
        # 输出文件
        self.layout_choose_out_file(layout)
        # 面积阈值
        area_layout = QHBoxLayout()
        self.area_spin = QSpinBox()
        self.area_spin.setRange(0, 100000000)
        self.area_spin.setValue(1000000)
        area_layout.addWidget(QLabel("小孔洞面积阈值:"))
        area_layout.addWidget(self.area_spin)
        layout.addLayout(area_layout)
        # 按钮
        self.create_button_layout(layout)
        self.setLayout(layout)

    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "area": self.area_spin.value()
        }
    
class ErodeDilateDialog(BaseDialog):
    def __init__(self, title="腐蚀操作参数设置", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        # 输入文件
        self.layout_choose_file(layout)
        # 输出文件
        self.layout_choose_out_file(layout)
        # kernel_size
        kernel_layout = QHBoxLayout()
        self.kernel_x = QSpinBox()
        self.kernel_x.setRange(1, 50)
        self.kernel_x.setValue(5)
        self.kernel_y = QSpinBox()
        self.kernel_y.setRange(1, 50)
        self.kernel_y.setValue(5)
        kernel_layout.addWidget(QLabel("Kernel Size (x,y):"))
        kernel_layout.addWidget(self.kernel_x)
        kernel_layout.addWidget(self.kernel_y)
        layout.addLayout(kernel_layout)
        # iterations
        iter_layout = QHBoxLayout()
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 50)
        self.iter_spin.setValue(1)
        iter_layout.addWidget(QLabel("Iterations:"))
        iter_layout.addWidget(self.iter_spin)
        layout.addLayout(iter_layout)
        # 按钮
        self.create_button_layout(layout)
        self.setLayout(layout)

    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "kernel_size": (self.kernel_x.value(), self.kernel_y.value()),
            "iterations": self.iter_spin.value()
        }

class SkeletonizeDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skeletonization 参数设置")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 输入文件
        self.layout_choose_file(layout)

        # 输出文件
        self.layout_choose_out_file(layout)

        # min_branch_length
        branch_layout = QHBoxLayout()
        self.branch_spin = QSpinBox()
        self.branch_spin.setRange(1, 100000000)
        self.branch_spin.setValue(1000000)
        branch_layout.addWidget(QLabel("最小分支长度:"))
        branch_layout.addWidget(self.branch_spin)
        layout.addLayout(branch_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        self.setLayout(layout)

    # ===== 获取参数 =====
    def get_params(self):
        return {
            "input_tif": self.input_tif.text(),
            "out_path": self.out_path.text(),
            "min_branch_length": self.branch_spin.value()
        }
