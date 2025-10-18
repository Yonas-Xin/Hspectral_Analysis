from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QTextEdit,
                             QSpinBox, QComboBox, QDialogButtonBox, QDoubleSpinBox,
                             QCheckBox)
from PyQt5.QtCore import Qt

import sys, os
import subprocess

def run_algorithm(algorithm, params):
    """
    在新终端中运行算法。
    
    algorithm: str, 算法名称，决定调用哪个脚本
    params: dict, 算法参数，需包含脚本所需的所有参数
    """

    # 根据算法类型组织命令行参数
    if algorithm == "random_crop":
        script_path = os.path.abspath("./toolbox_win/function1_rci.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--output_dir", params["output_dir"],
            "--patch_size", str(params["patch_size"]),
            "--sample_fraction", str(params["sample_fraction"]),
            "--image_block", str(params["image_block"])
        ]

    elif algorithm == "superpixel_sampling":
        script_path = os.path.abspath("./toolbox_win/function2_1_ss.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--out_shp", params["out_shp"],
            "--max_samples", str(params["max_samples"]),
            "--n_segments", str(params["n_segments"]),
            "--enhance_func", params["enhance_func"],
            "--embedding_nums", str(params["embedding_nums"]),
            "--compactness", str(params["compactness"]),
            "--ppi_niters", str(params["ppi_niters"]),
            "--ppi_threshold", str(params["ppi_threshold"]),
            "--ppi_centered", str(params["ppi_centered"])
        ]

    # elif algorithm == "smacc_sampling":
    #     script_path = os.path.abspath("./toolbox_win/function2_2_smaccs.py")
    #     cmd = [
    #         sys.executable, script_path,
    #         "--input_tif", params["input_tif"],
    #         "--output_dir", params["output_dir"],
    #         "--row", str(params["row"]),
    #         "--col", str(params["col"]),
    #         "--embedding_nums", str(params["embedding_nums"]),
    #         "--samples", str(params["samples"])
    #     ]

    elif algorithm == "dd_prediction":
        script_path = os.path.abspath("./toolbox_win/function3_dd.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--input_shp", params["input_shp"],
            "--output_csv", params["output_csv"],
            "--FUNC", params["FUNC"],
            "--patch_size", str(params["patch_size"]),
            "--batch_size", str(params["batch_size"]),
            "--embedding_dims", str(params["embedding_dims"])
        ]
        cmd += ["--model_path", str(params["model_path"])]

    elif algorithm == "cluster_features":
        script_path = os.path.abspath("./toolbox_win/function4_c.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--input_shp", params["input_shp"],
            "--feature_csv", params["feature_csv"],
            "--n_clusters", str(params["n_clusters"]),
            "--output_dir", params["output_dir"],
            "--output_dir_name", params.get("output_dir_name", "SAMPLES_DIR")
        ]

    elif algorithm == "sample_optimize":
        script_path = os.path.abspath("./toolbox_win/function5_so.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--input_shp_dir", params["input_shp_dir"],
            "--data_form", params["data_form"],
            "--FUNC", params["FUNC"],
            "--ratio", str(params["ratio"]),
            "--n_splits", str(params["n_splits"])
        ]
        cmd += ["--embedding_csv", str(params["embedding_csv"])]
    
    elif algorithm == "sample_crop":
        script_path = os.path.abspath("./toolbox_win/function6_sc.py")
        cmd = [
            sys.executable, script_path,
            "--input_tif", params["input_tif"],
            "--input_shp_dir", params["input_shp_dir"],
            "--out_dir", params["out_dir"],
            "--patch_size", str(params["patch_size"]),
            "--out_tif_name", params["out_tif_name"]
        ]

    else:
        raise ValueError(f"未知算法类型: {algorithm}")

    # 在新终端窗口中执行（Windows）
    subprocess.Popen(["start", "cmd", "/k"] + cmd, shell=True)


class BaseDialog(QDialog):
    """基础对话框类"""
    def __init__(self, parent=None, title="参数设置"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(600, 400)
        
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
    
    def layout_choose_out_dir(self, layout):
        """创建选择输出目录的布局"""
        out_layout = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("请选择输出目录...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_out_dir)
        out_layout.addWidget(QLabel("输出目录:"))
        out_layout.addWidget(self.out_dir)
        out_layout.addWidget(browse_btn)
        layout.addLayout(out_layout)
        return out_layout

    def create_label_layout(self, layout, text):
        self.info_label = QLabel("算法信息：")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setText(text)
        layout.addWidget(self.info_label)
        layout.addWidget(self.info_text)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "Data Files (*.tif *.dat)")
        if file_path:
            self.input_tif.setText(file_path)

    def browse_out_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出文件", "", "TIFF Files (*.tif)")
        if file_path:
            self.out_path.setText(file_path)
    
    def browse_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.out_dir.setText(dir_path)

class RandomCropDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "随机裁剪参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输出目录
        self.layout_choose_out_dir(layout)

        # patch_size
        patch_layout = QHBoxLayout()
        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(1, 512)
        self.patch_spin.setValue(17)
        patch_layout.addWidget(QLabel("patch_size:"))
        patch_layout.addWidget(self.patch_spin)
        layout.addLayout(patch_layout)

        # sample_fraction
        fraction_layout = QHBoxLayout()
        self.fraction_spin = QDoubleSpinBox()
        self.fraction_spin.setRange(0.000001, 1.0)
        self.fraction_spin.setDecimals(4)
        self.fraction_spin.setSingleStep(0.001)
        self.fraction_spin.setValue(0.001)
        fraction_layout.addWidget(QLabel("sample_fraction:"))
        fraction_layout.addWidget(self.fraction_spin)
        layout.addLayout(fraction_layout)

        # image_block
        block_layout = QHBoxLayout()
        self.block_spin = QSpinBox()
        self.block_spin.setRange(16, 4096)
        self.block_spin.setValue(512)
        block_layout.addWidget(QLabel("image_block:"))
        block_layout.addWidget(self.block_spin)
        layout.addLayout(block_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "该算法对高光谱影像进行随机采样与裁剪，"
            "生成用于对比学习预训练的小块样本。\n\n"
            "参数说明:\n"
            "1. 输入文件: 待处理的高光谱影像(*tif、*dat)\n"
            "2. 输出目录: 样本保存位置\n"
            "3. patch_size: 裁剪块的大小（像素）\n"
            "4. sample_fraction: 随机采样比例，控制样本数量\n"
            "5. image_block: 分块大小，决定处理时的内存占用"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "output_dir": self.out_dir.text(),
            "patch_size": self.patch_spin.value(),
            "sample_fraction": self.fraction_spin.value(),
            "image_block": self.block_spin.value()
        }
    
class SuperpixelSamplingDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "超像素采样参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输出 shp
        out_layout = QHBoxLayout()
        self.out_shp = QLineEdit()
        self.out_shp.setPlaceholderText("请选择输出shp文件路径...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_out_shp)
        out_layout.addWidget(QLabel("输出矢量:"))
        out_layout.addWidget(self.out_shp)
        out_layout.addWidget(browse_btn)
        layout.addLayout(out_layout)

        # max_samples
        samples_layout = QHBoxLayout()
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(1, 10000)
        self.max_samples_spin.setValue(30)
        samples_layout.addWidget(QLabel("max_samples:"))
        samples_layout.addWidget(self.max_samples_spin)
        layout.addLayout(samples_layout)

        # n_segments
        seg_layout = QHBoxLayout()
        self.seg_spin = QSpinBox()
        self.seg_spin.setRange(2, 10000)
        self.seg_spin.setValue(512)
        seg_layout.addWidget(QLabel("n_segments:"))
        seg_layout.addWidget(self.seg_spin)
        layout.addLayout(seg_layout)

        # enhance_func
        enhance_layout = QHBoxLayout()
        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems(['MNF', 'PCA'])
        self.enhance_combo.setCurrentText('MNF')
        enhance_layout.addWidget(QLabel("enhance_func:"))
        enhance_layout.addWidget(self.enhance_combo)
        layout.addLayout(enhance_layout)

        # embedding_nums
        emb_layout = QHBoxLayout()
        self.emb_spin = QSpinBox()
        self.emb_spin.setRange(1, 100)
        self.emb_spin.setValue(12)
        emb_layout.addWidget(QLabel("embedding_nums:"))
        emb_layout.addWidget(self.emb_spin)
        layout.addLayout(emb_layout)

        # compactness
        comp_layout = QHBoxLayout()
        self.comp_spin = QSpinBox()
        self.comp_spin.setRange(1, 1000)
        self.comp_spin.setValue(25)
        comp_layout.addWidget(QLabel("compactness:"))
        comp_layout.addWidget(self.comp_spin)
        layout.addLayout(comp_layout)

        # ppi_niters
        niters_layout = QHBoxLayout()
        self.niters_spin = QSpinBox()
        self.niters_spin.setRange(1, 100000)
        self.niters_spin.setValue(2000)
        niters_layout.addWidget(QLabel("ppi_niters:"))
        niters_layout.addWidget(self.niters_spin)
        layout.addLayout(niters_layout)

        # ppi_threshold
        threshold_layout = QHBoxLayout()
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(0)
        threshold_layout.addWidget(QLabel("ppi_threshold:"))
        threshold_layout.addWidget(self.threshold_spin)
        layout.addLayout(threshold_layout)

        # ppi_centered
        self.ppi_check = QCheckBox("ppi_centered")
        self.ppi_check.setChecked(False)
        layout.addWidget(self.ppi_check)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "使用超像素分割slic+ppi端元识别的采样方法，对输入高光谱影像采取代表性端元，"
            "生成用于后续处理的采样点矢量。\n\n"
            "参数说明:\n"
            "1. 输入文件: 待处理的高光谱影像(*tif、*dat)\n"
            "2. 输出矢量: 结果保存的shp文件路径\n"
            "3. max_samples: 最大采样点数\n"
            "4. n_segments: 超像素分割数\n"
            "5. enhance_func: 特征增强方法 (MNF, PCA, ICA)\n"
            "6. embedding_nums: 降维嵌入维度数\n"
            "7. compactness: 超像素紧致度参数\n"
            "8. ppi_niters: PPI迭代次数\n"
            "9. ppi_threshold: PPI阈值\n"
            "10. ppi_centered: 是否启用居中化处理"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_out_shp(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出矢量文件", "", "Shapefiles (*.shp)")
        if file_path:
            self.out_shp.setText(file_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "out_shp": self.out_shp.text(),
            "max_samples": self.max_samples_spin.value(),
            "n_segments": self.seg_spin.value(),
            "enhance_func": self.enhance_combo.currentText(),
            "embedding_nums": self.emb_spin.value(),
            "compactness": self.comp_spin.value(),
            "ppi_niters": self.niters_spin.value(),
            "ppi_threshold": self.threshold_spin.value(),
            "ppi_centered": self.ppi_check.isChecked()
        }


class SmaccSamplingDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "SMACC采样参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输出 shp
        out_layout = QHBoxLayout()
        self.out_shp = QLineEdit()
        self.out_shp.setPlaceholderText("请选择输出shp文件路径...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_out_shp)
        out_layout.addWidget(QLabel("输出矢量:"))
        out_layout.addWidget(self.out_shp)
        out_layout.addWidget(browse_btn)
        layout.addLayout(out_layout)

        # row
        row_layout = QHBoxLayout()
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 100)
        self.row_spin.setValue(3)
        row_layout.addWidget(QLabel("row:"))
        row_layout.addWidget(self.row_spin)
        layout.addLayout(row_layout)

        # col
        col_layout = QHBoxLayout()
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 100)
        self.col_spin.setValue(3)
        col_layout.addWidget(QLabel("col:"))
        col_layout.addWidget(self.col_spin)
        layout.addLayout(col_layout)

        # embedding_nums
        emb_layout = QHBoxLayout()
        self.emb_spin = QSpinBox()
        self.emb_spin.setRange(1, 100)
        self.emb_spin.setValue(12)
        emb_layout.addWidget(QLabel("embedding_nums:"))
        emb_layout.addWidget(self.emb_spin)
        layout.addLayout(emb_layout)

        # samples
        samples_layout = QHBoxLayout()
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 1000000)
        self.samples_spin.setValue(4000)
        samples_layout.addWidget(QLabel("samples:"))
        samples_layout.addWidget(self.samples_spin)
        layout.addLayout(samples_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "分块SMACC (Sequential Maximum Angle Convex Cone) 采样方法，"
            "通过划分子区域，构建高光谱数据的端元空间，提取具有代表性的样本。\n\n"
            "参数说明:\n"
            "1. 输入文件: 待处理的高光谱影像(*tif、*dat)\n"
            "2. 输出目录: 样本保存位置\n"
            "3. row: 分块行数，用于划分子区域\n"
            "4. col: 分块列数，用于划分子区域\n"
            "5. embedding_nums: 降维后的嵌入维度\n"
            "6. samples: 采样点数量"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_out_shp(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出矢量文件", "", "Shapefiles (*.shp)")
        if file_path:
            self.out_shp.setText(file_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "output_dir": self.out_dir.text(),
            "row": self.row_spin.value(),
            "col": self.col_spin.value(),
            "embedding_nums": self.emb_spin.value(),
            "samples": self.samples_spin.value()
        }

class DdPredictionDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "特征降维参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输入矢量
        shp_layout = QHBoxLayout()
        self.input_shp = QLineEdit()
        self.input_shp.setPlaceholderText("请选择输入矢量文件...")
        browse_shp_btn = QPushButton("浏览")
        browse_shp_btn.clicked.connect(self.browse_shp_file)
        shp_layout.addWidget(QLabel("输入矢量:"))
        shp_layout.addWidget(self.input_shp)
        shp_layout.addWidget(browse_shp_btn)
        layout.addLayout(shp_layout)

        # 输出CSV
        csv_layout = QHBoxLayout()
        self.output_csv = QLineEdit()
        self.output_csv.setPlaceholderText("输出CSV文件路径...")
        browse_csv_btn = QPushButton("浏览")
        browse_csv_btn.clicked.connect(self.browse_csv_file)
        csv_layout.addWidget(QLabel("输出CSV:"))
        csv_layout.addWidget(self.output_csv)
        csv_layout.addWidget(browse_csv_btn)
        layout.addLayout(csv_layout)

        # FUNC
        func_layout = QHBoxLayout()
        self.func_combo = QComboBox()
        self.func_combo.addItems(["DL", "PCA", "MNF"])  # 举例添加几个常见分类器
        self.func_combo.setCurrentText("DL")
        func_layout.addWidget(QLabel("FUNC:"))
        func_layout.addWidget(self.func_combo)
        layout.addLayout(func_layout)

        # 模型路径（可选）
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("可选：加载已有模型路径...")
        browse_model_btn = QPushButton("浏览")
        browse_model_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(browse_model_btn)
        layout.addLayout(model_layout)

        # patch_size
        patch_layout = QHBoxLayout()
        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(1, 512)
        self.patch_spin.setValue(17)
        patch_layout.addWidget(QLabel("patch_size:"))
        patch_layout.addWidget(self.patch_spin)
        layout.addLayout(patch_layout)

        # batch_size
        batch_layout = QHBoxLayout()
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 10000)
        self.batch_spin.setValue(256)
        batch_layout.addWidget(QLabel("batch_size:"))
        batch_layout.addWidget(self.batch_spin)
        layout.addLayout(batch_layout)

        # embedding_dims
        emb_layout = QHBoxLayout()
        self.emb_spin = QSpinBox()
        self.emb_spin.setRange(1, 512)
        self.emb_spin.setValue(24)
        emb_layout.addWidget(QLabel("embedding_dims:"))
        emb_layout.addWidget(self.emb_spin)
        layout.addLayout(emb_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "特征降维算法对输入的高光谱影像及矢量样本进行特征提取，"
            "支持多种降维方法（如DL、SVM、RF），并输出降维编码结果CSV。\n\n"
            "参数说明:\n"
            "1. 输入文件: 高光谱影像(*tif、*dat)\n"
            "2. 输入矢量: 矢量样本文件(*shp)\n"
            "3. 输出CSV: 保存预测结果的文件路径\n"
            "4. FUNC: 分类方法（DL, SVM, RF 等）\n"
            "5. 模型路径: 可选，加载已有训练模型，当FUNC为DL时必须提供\n"
            "6. patch_size: 裁剪块大小（像素）\n"
            "7. batch_size: 批量大小，影响运行速度和显存\n"
            "8. embedding_dims: 特征嵌入维度"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_shp_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入矢量文件", "", "Shapefiles (*.shp)")
        if file_path:
            self.input_shp.setText(file_path)

    def browse_csv_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.output_csv.setText(file_path)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pt)")
        if file_path:
            self.model_path.setText(file_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "input_shp": self.input_shp.text(),
            "output_csv": self.output_csv.text(),
            "FUNC": self.func_combo.currentText(),
            "model_path": self.model_path.text() if self.model_path.text() else None,
            "patch_size": self.patch_spin.value(),
            "batch_size": self.batch_spin.value(),
            "embedding_dims": self.emb_spin.value()
        }


class ClusterFeaturesDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "样本聚类参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输入矢量
        shp_layout = QHBoxLayout()
        self.input_shp = QLineEdit()
        self.input_shp.setPlaceholderText("请选择输入矢量文件...")
        browse_shp_btn = QPushButton("浏览")
        browse_shp_btn.clicked.connect(self.browse_shp_file)
        shp_layout.addWidget(QLabel("输入矢量:"))
        shp_layout.addWidget(self.input_shp)
        shp_layout.addWidget(browse_shp_btn)
        layout.addLayout(shp_layout)

        # 特征CSV
        csv_layout = QHBoxLayout()
        self.feature_csv = QLineEdit()
        self.feature_csv.setPlaceholderText("请选择特征CSV文件...")
        browse_csv_btn = QPushButton("浏览")
        browse_csv_btn.clicked.connect(self.browse_csv_file)
        csv_layout.addWidget(QLabel("特征CSV:"))
        csv_layout.addWidget(self.feature_csv)
        csv_layout.addWidget(browse_csv_btn)
        layout.addLayout(csv_layout)

        # 聚类数
        cluster_layout = QHBoxLayout()
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(1, 1000)
        self.cluster_spin.setValue(10)
        cluster_layout.addWidget(QLabel("n_clusters:"))
        cluster_layout.addWidget(self.cluster_spin)
        layout.addLayout(cluster_layout)

        # 输出目录
        self.layout_choose_out_dir(layout)

        # 输出目录名
        outname_layout = QHBoxLayout()
        self.out_name = QLineEdit("SAMPLES_DIR")
        outname_layout.addWidget(QLabel("output_dir_name:"))
        outname_layout.addWidget(self.out_name)
        layout.addLayout(outname_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "该算法对输入的样本进行自动聚类，结合矢量样本和特征CSV，算法默认为GMM聚类，"
            "生成聚类结果以shp格式存储，每个shp文件都是一个聚类结果点位，并输出到指定目录。\n\n"
            "参数说明:\n"
            "1. 输入文件: 高光谱影像(*tif、*dat)\n"
            "2. 输入矢量: 矢量样本文件(*shp)\n"
            "3. 特征CSV: 提取的特征文件\n"
            "4. n_clusters: 聚类的类别数\n"
            "5. 输出目录: 结果保存位置\n"
            "6. output_dir_name: 输出子目录名，默认 SAMPLES_DIR"
        )
        self.create_label_layout(info_layout, text)

        # 合并布局
        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_shp_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入矢量文件", "", "Shapefiles (*.shp)")
        if file_path:
            self.input_shp.setText(file_path)

    def browse_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择特征CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.feature_csv.setText(file_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "input_shp": self.input_shp.text(),
            "feature_csv": self.feature_csv.text() if self.feature_csv.text() else None,
            "n_clusters": self.cluster_spin.value(),
            "output_dir": self.out_dir.text(),
            "output_dir_name": self.out_name.text()
        }


class SampleOptimizeDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "样本优化参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输入矢量目录
        shp_layout = QHBoxLayout()
        self.input_shp_dir = QLineEdit()
        self.input_shp_dir.setPlaceholderText("请选择输入矢量目录...")
        browse_shp_btn = QPushButton("浏览")
        browse_shp_btn.clicked.connect(self.browse_shp_dir)
        shp_layout.addWidget(QLabel("矢量目录:"))
        shp_layout.addWidget(self.input_shp_dir)
        shp_layout.addWidget(browse_shp_btn)
        layout.addLayout(shp_layout)

        # data_form
        data_layout = QHBoxLayout()
        self.data_combo = QComboBox()
        self.data_combo.addItems(["Spectral", "Embedding"])
        self.data_combo.setCurrentText("Spectral")
        data_layout.addWidget(QLabel("data_form:"))
        data_layout.addWidget(self.data_combo)
        layout.addLayout(data_layout)

        # embedding_csv
        emb_layout = QHBoxLayout()
        self.embedding_csv = QLineEdit()
        self.embedding_csv.setPlaceholderText("可选：Embedding CSV 文件路径")
        browse_csv_btn = QPushButton("浏览")
        browse_csv_btn.clicked.connect(self.browse_embedding_csv)
        emb_layout.addWidget(QLabel("embedding_csv:"))
        emb_layout.addWidget(self.embedding_csv)
        emb_layout.addWidget(browse_csv_btn)
        layout.addLayout(emb_layout)

        # FUNC
        func_layout = QHBoxLayout()
        self.func_combo = QComboBox()
        self.func_combo.addItems(["LF", "Mccv_LF"])
        self.func_combo.setCurrentText("LF")
        func_layout.addWidget(QLabel("FUNC:"))
        func_layout.addWidget(self.func_combo)
        layout.addLayout(func_layout)

        # ratio
        ratio_layout = QHBoxLayout()
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.0, 1.0)
        self.ratio_spin.setSingleStep(0.01)
        self.ratio_spin.setValue(0.2)
        ratio_layout.addWidget(QLabel("ratio:"))
        ratio_layout.addWidget(self.ratio_spin)
        layout.addLayout(ratio_layout)

        # n_splits
        split_layout = QHBoxLayout()
        self.splits_spin = QSpinBox()
        self.splits_spin.setRange(1, 100000)
        self.splits_spin.setValue(1000)
        split_layout.addWidget(QLabel("n_splits:"))
        split_layout.addWidget(self.splits_spin)
        layout.addLayout(split_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "样本优化算法对高光谱影像和对应矢量样本进行最优样本选择，按照比例剔除冗余样本，"
            "可使用光谱或特征编码形式数据，并支持不同优化方法。\n\n"
            "参数说明:\n"
            "1. 输入文件: 高光谱影像(*tif、*dat)\n"
            "2. 输入矢量目录: 保存矢量样本文件的目录\n"
            "3. data_form: 数据形式（Spectral 或 Embedding）\n"
            "4. embedding_csv: 可选，嵌入特征CSV文件路径，当数据形式为Embedding时必须提供\n"
            "5. FUNC: 优化方法 (LF 或 Mccv_LF)\n"
            "6. ratio: 样本划分比例\n"
            "7. n_splits: 划分次数或迭代次数，仅当FUNC为Mccv_LF时有效"
        )
        self.create_label_layout(info_layout, text)

        # 合并布局
        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_shp_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入矢量目录")
        if dir_path:
            self.input_shp_dir.setText(dir_path)

    def browse_embedding_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择Embedding CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.embedding_csv.setText(file_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "input_shp_dir": self.input_shp_dir.text(),
            "data_form": self.data_combo.currentText(),
            "embedding_csv": self.embedding_csv.text() if self.embedding_csv.text() else None,
            "FUNC": self.func_combo.currentText(),
            "ratio": self.ratio_spin.value(),
            "n_splits": self.splits_spin.value()
        }


class SampleCropDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "矢量裁剪参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入影像
        self.layout_choose_file(layout)

        # 输入 shp 文件夹
        shp_layout = QHBoxLayout()
        self.shp_dir = QLineEdit()
        self.shp_dir.setPlaceholderText("请选择输入shp文件夹...")
        browse_shp_btn = QPushButton("浏览")
        browse_shp_btn.clicked.connect(self.browse_shp_dir)
        shp_layout.addWidget(QLabel("输入shp文件夹:"))
        shp_layout.addWidget(self.shp_dir)
        shp_layout.addWidget(browse_shp_btn)
        layout.addLayout(shp_layout)

        # 输出目录
        self.layout_choose_out_dir(layout)

        # patch_size
        patch_layout = QHBoxLayout()
        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(1, 512)
        self.patch_spin.setValue(17)
        patch_layout.addWidget(QLabel("patch_size:"))
        patch_layout.addWidget(self.patch_spin)
        layout.addLayout(patch_layout)

        # out_tif_name
        out_name_layout = QHBoxLayout()
        self.out_name_edit = QLineEdit("IMG_PATCH")
        out_name_layout.addWidget(QLabel("输出影像前缀:"))
        out_name_layout.addWidget(self.out_name_edit)
        layout.addLayout(out_name_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "该算法根据提供的矢量shp文件对栅格影像进行裁剪，每一个类别应该用一个单独的shp文件进行存储"
            "生成以矢量范围为裁剪区域的图像块。\n\n"
            "参数说明:\n"
            "1. 输入文件: 待裁剪的栅格影像 (*tif)\n"
            "2. 输入 shp 文件夹: 存放裁剪矢量文件的目录\n"
            "3. 输出目录: 保存裁剪结果的路径\n"
            "4. patch_size: 裁剪图像块大小（像素）\n"
            "5. 输出影像前缀: 裁剪结果的命名"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_shp_dir(self):
        """选择 shp 文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入 shp 文件夹", "")
        if dir_path:
            self.shp_dir.setText(dir_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_tif": self.input_tif.text(),
            "input_shp_dir": self.shp_dir.text(),
            "out_dir": self.out_dir.text(),
            "patch_size": self.patch_spin.value(),
            "out_tif_name": self.out_name_edit.text()
        }

class SampleSplitDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "样本分割参数设置")
        self.initUI()

    def initUI(self):
        # 总体水平布局：左边参数，右边信息
        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        # 输入 shp 文件夹
        in_shp_layout = QHBoxLayout()
        self.input_shp_dir = QLineEdit()
        self.input_shp_dir.setPlaceholderText("请选择输入 shp 文件夹...")
        browse_in_btn = QPushButton("浏览")
        browse_in_btn.clicked.connect(self.browse_input_shp_dir)
        in_shp_layout.addWidget(QLabel("输入 shp 文件夹:"))
        in_shp_layout.addWidget(self.input_shp_dir)
        in_shp_layout.addWidget(browse_in_btn)
        layout.addLayout(in_shp_layout)

        # 输出 shp 文件夹
        out_shp_layout = QHBoxLayout()
        self.output_shp_dir = QLineEdit()
        self.output_shp_dir.setPlaceholderText("请选择输出 shp 文件夹...")
        browse_out_btn = QPushButton("浏览")
        browse_out_btn.clicked.connect(self.browse_output_shp_dir)
        out_shp_layout.addWidget(QLabel("输出 shp 文件夹:"))
        out_shp_layout.addWidget(self.output_shp_dir)
        out_shp_layout.addWidget(browse_out_btn)
        layout.addLayout(out_shp_layout)

        # num_to_select
        num_layout = QHBoxLayout()
        self.num_spin = QDoubleSpinBox()
        self.num_spin.setRange(0.0, 1e6)
        self.num_spin.setDecimals(3)
        self.num_spin.setValue(0.6)
        num_layout.addWidget(QLabel("num_to_select:"))
        num_layout.addWidget(self.num_spin)
        layout.addLayout(num_layout)

        # 确认/取消按钮
        self.create_button_layout(layout)

        # 右边信息展示
        info_layout = QVBoxLayout()
        text = (
            "算法概述:\n"
            "该算法对输入的 shp 文件夹内的样本要素进行随机选择，分别划分到训练集与验证集"
            "并将结果保存到新的 shp 文件夹。\n\n"
            "参数说明:\n"
            "1. 输入 shp 文件夹: 待处理的矢量文件目录\n"
            "2. 输出 shp 文件夹: 保存结果的目录\n"
            "3. num_to_select: 随机选择的要素数量。\n"
            "   - 如果大于 1，则表示选择的绝对数量。\n"
            "   - 如果小于等于 1，则按比例选择。"
        )
        self.create_label_layout(info_layout, text)

        main_layout.addLayout(layout, 3)
        main_layout.addLayout(info_layout, 2)
        self.setLayout(main_layout)

    def browse_input_shp_dir(self):
        """选择输入 shp 文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入 shp 文件夹", "")
        if dir_path:
            self.input_shp_dir.setText(dir_path)

    def browse_output_shp_dir(self):
        """选择输出 shp 文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出 shp 文件夹", "")
        if dir_path:
            self.output_shp_dir.setText(dir_path)

    def get_params(self):
        """返回用户输入的参数字典"""
        return {
            "input_shp_dir": self.input_shp_dir.text(),
            "output_shp_dir": self.output_shp_dir.text(),
            "num_to_select": self.num_spin.value()
        }
