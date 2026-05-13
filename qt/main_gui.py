import sys
import os

# 修复 PyInstaller -w 模式打包后，sys.stdout/stderr 为 None 导致的 tqdm 及 print 报错 ('NoneType' object has no attribute 'write')
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import copy
import tempfile
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QFileDialog, QMessageBox, QListWidget,
                             QAbstractItemView, QGroupBox, QSplitter, QFormLayout,
                             QListWidgetItem, QScrollArea, QProgressBar, QListView)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# 将上一级目录加入path以便导入 modules
if getattr(sys, 'frozen', False):
    # 如果是用 PyInstaller 打包后的 exe 环境
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(bundle_dir)
import modules.Augmentation_CV as Augmentation_CV
import modules.Augmentation_AL as Augmentation_AL
import modules.background as background
import modules.image_mask_AL as image_mask_AL
import modules.shift_classification as shift_classification
import modules.shift_detection as shift_detection
import modules.image_mask as image_mask

OPERATIONS_TEMPLATE = {
    "1. OpenCV - 尺寸调整/裁剪": {
        "kwargs": {"mode": 'stretch', "w": 128, "h": 128}
    },
    "2. OpenCV - 像素化": {
        "kwargs": {"size": 3}
    },
    "3. Albumentations - 数据增强": {
        "kwargs": {"num_augments": 5}
    },
    "4. Image Mask - 背景合成 (需背景文件夹)": {
        "kwargs": {"min_scale": 0.7, "max_scale": 1.1, "min_visible": 0.9, "num_augments": 6}
    },
    "5. OpenCV - 局部光照": {
        "kwargs": {"strength_range_min": -40, "strength_range_max": 80, 
                   "radius_range_min": 50, "radius_range_max": 240, "num_spots": 4}  # 拆解元组
    },
    "6. 分类 - 数据集划分": {
        "kwargs": {"train_ratio": 0.75, "val_ratio": 0.2, "test_ratio": 0.05, "seed": 749}
    },
    "7. Background - 生成噪声背景图": {
        "kwargs": {"size_h": 224, "size_w": 224, "white_background": False, "noise_density": 0.05}
    },
    "8. Image Mask - 简单拼贴背景合成": {
        "kwargs": {"min_scale": 0.8, "max_scale": 1.2, "min_visible": 0.8, "center_mode": False}
    },
    "9. 检测 - YOLO数据集划分": {
        "kwargs": {"train_ratio": 0.8, "copy_files": True, "random_seed": 42}
    }
}

class PipelineListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setStyleSheet("""
            QListWidget { 
                background-color: #FAFAFA; 
                border: 2px dashed #DCDFE6; 
                border-radius: 8px; 
                padding: 5px;
            }
            QListWidget::item {
                background-color: #FFFFFF;
                border: 1px solid #E4E7ED;
                border-radius: 5px;
                padding: 8px;
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: #ECF5FF;
                border: 1px solid #409EFF;
                color: #409EFF;
            }
        """)

    def dropEvent(self, event):
        if event.source() == self:
            # 内部顺序拖拽移动
            event.setDropAction(Qt.MoveAction)
            super().dropEvent(event)
        elif event.source():
            # 外部 (如组件库) 拖拽拉入
            item = event.source().currentItem()
            if item:
                op_name = item.text()
                new_item = QListWidgetItem(op_name)
                # 为该条目赋初值
                new_item.setData(Qt.UserRole, copy.deepcopy(OPERATIONS_TEMPLATE[op_name]["kwargs"]))
                
                # 插入到光标处或添加到末尾
                index = self.indexAt(event.pos())
                if index.isValid():
                    self.insertItem(index.row(), new_item)
                else:
                    self.addItem(new_item)
                
                self.setCurrentItem(new_item)
                event.accept()

def run_op(op_name, kwargs, cur_in, cur_out, bg_dir):
    """独立的流水线执行函数"""
    
    # if not (op_name.startswith("1.") or op_name.startswith("6.") or op_name.startswith("8.")):
    #     raise ImportError(f"DLL load failed while importing _core_engine for module {op_name.split(' - ')[0]}: The specified component requires complete dependencies which are missing in this environment.")

    if op_name.startswith("1."):
        Augmentation_CV.batch_yasuo(cur_in, cur_out, **kwargs)
    elif op_name.startswith("2."):
        Augmentation_CV.batch_pixelate(cur_in, cur_out, **kwargs)
    elif op_name.startswith("3."):
        Augmentation_AL.augment_images(cur_in, cur_out, **kwargs)
    elif op_name.startswith("4."):
        if not bg_dir:
            raise ValueError("背景图文件夹未设置")
        image_mask_AL.batch_overlay(bg_dir, cur_in, cur_out, **kwargs)
    elif op_name.startswith("5."):
        # 组装回去
        real_kwargs = {
            "strength_range": (kwargs["strength_range_min"], kwargs["strength_range_max"]),
            "radius_range": (kwargs["radius_range_min"], kwargs["radius_range_max"]),
            "num_spots": kwargs["num_spots"]
        }
        Augmentation_CV.batch_local_illumination(cur_in, cur_out, **real_kwargs)
    elif op_name.startswith("6."):
        shift_classification.split_dataset(cur_in, cur_out, **kwargs)
    elif op_name.startswith("7."):
        # 这一步比较特殊，它不需要输入文件夹的图片，但是为了流水线跑通我们在输出文件夹生成
        real_size = (kwargs["size_h"], kwargs["size_w"])
        save_file = os.path.join(cur_out, "generated_noise_bg.png")
        background.generate_noise_image(size=real_size, 
                                        save_path=save_file, 
                                        white_background=kwargs["white_background"], 
                                        noise_density=kwargs["noise_density"], 
                                        show=False)
    elif op_name.startswith("8."):
        if not bg_dir:
            raise ValueError("背景图文件夹未设置")
        image_mask.batch_overlay(bg_dir, cur_in, cur_out, **kwargs)
    elif op_name.startswith("9."):
        # 注意: 这里的cur_in必须是yolo格式，包含images和labels文件夹
        shift_detection.split_yolo_dataset(cur_in, **kwargs)
    else:
        raise NotImplementedError(f"未实现的操作: {op_name}")

class PipelineWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, steps, input_dir, output_dir, bg_dir):
        super().__init__()
        self.steps = steps
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.bg_dir = bg_dir

    def run(self):
        try:
            current_input = self.input_dir
            for i, step in enumerate(self.steps):
                op_name = step['op_name']
                kwargs = step['kwargs']
                
                safe_name = op_name.replace(" ", "").replace("/", "_")
                step_out = os.path.join(self.output_dir, f"step{i}_{safe_name}")
                if not os.path.exists(step_out):
                    os.makedirs(step_out)
                
                run_op(op_name, kwargs, current_input, step_out, self.bg_dir)
                
                current_input = step_out
                self.progress_signal.emit(i + 1, op_name)
                
            self.finished_signal.emit(current_input)
        except Exception as e:
            self.error_signal.emit(str(e))

class AugmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Augmentation Pipeline Tool")
        # 预设一个足够大的基础尺寸，防止 Windows DPI 缩放下的假最大化 Bug
        self.resize(1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. 顶部：全局路径设置与组件库
        path_group = QGroupBox("全局资源与功能组件库")
        # 稍微增加高度以容纳列表
        path_group.setMaximumHeight(160)
        path_layout = QHBoxLayout()
        
        left_paths = QVBoxLayout()
        # 依照要求：背景文件夹 -> 原始图文件夹 -> 输出文件夹
        self.bg_edit, btn_bg = self.create_path_row(left_paths, "背景文件夹:")
        btn_bg.clicked.connect(lambda: self.browse_folder(self.bg_edit))
        
        self.input_edit, btn_in = self.create_path_row(left_paths, "原始图文件夹:")
        btn_in.clicked.connect(lambda: self.browse_folder(self.input_edit))
        
        self.output_edit, btn_out = self.create_path_row(left_paths, "输出文件夹:")
        btn_out.clicked.connect(lambda: self.browse_folder(self.output_edit))
        
        # 将原右侧路径替换为功能组件库，模拟 AD 拖放
        right_palette = QVBoxLayout()
        right_palette.addWidget(QLabel("功能组件库:"))
        self.palette = QListWidget()
        self.palette.setDragEnabled(True)
        self.palette.setDragDropMode(QAbstractItemView.DragOnly)
        self.palette.setDefaultDropAction(Qt.CopyAction)
        self.palette.setViewMode(QListView.IconMode)
        self.palette.setFlow(QListView.LeftToRight)
        self.palette.setSpacing(10)
        self.palette.setResizeMode(QListView.Adjust)
        self.palette.setStyleSheet("""
            QListWidget { 
                background-color: transparent; 
                border: none; 
            }
            QListWidget::item { 
                background-color: #FFFFFF; 
                border: 1px solid #DCDFE6; 
                border-radius: 20px; 
                padding: 5px 15px; 
                margin: 2px;
            }
            QListWidget::item:hover { 
                background-color: #ECF5FF; 
                border: 1px solid #409EFF; 
                color: #409EFF;
            }
        """)
        
        for op in OPERATIONS_TEMPLATE.keys():
            item = QListWidgetItem(op)
            item.setTextAlignment(Qt.AlignCenter)
            item.setSizeHint(QSize(180, 40))
            self.palette.addItem(item)
            
        right_palette.addWidget(self.palette)
        
        path_layout.addLayout(left_paths, 4)
        path_layout.addLayout(right_palette, 6)
        path_group.setLayout(path_layout)
        # stretch系数0代表不拉伸，使其保持本来大小
        main_layout.addWidget(path_group, 0)

        # 2. 中下部分结构 (水平分隔)
        splitter = QSplitter(Qt.Horizontal)
        # stretch系数1代表占据剩余的主要空间
        main_layout.addWidget(splitter, 1)

        # 2.1 左侧：当前模块参数配置
        left_panel = QGroupBox("模块参数设置")
        left_vbox = QVBoxLayout()
        
        # 滚动区域以防参数过多
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_widget = QWidget()
        self.param_layout = QFormLayout(self.param_widget)
        self.param_scroll.setWidget(self.param_widget)
        
        left_vbox.addWidget(QLabel("选择流水线中的模块以修改参数"))
        left_vbox.addWidget(self.param_scroll)
        left_panel.setLayout(left_vbox)
        splitter.addWidget(left_panel)

        # 2.2 中间：效果预览区域
        center_panel = QGroupBox("预览")
        center_vbox = QVBoxLayout()
        
        image_layout = QHBoxLayout()
        
        self.ori_label = QLabel("原图加载中...")
        self.ori_label.setAlignment(Qt.AlignCenter)
        self.ori_label.setStyleSheet("background-color: #EBEEF5; color: #909399; border: 1px dashed #DCDFE6; border-radius: 8px;")
        
        self.res_label = QLabel("运行预览后显示结果")
        self.res_label.setAlignment(Qt.AlignCenter)
        self.res_label.setStyleSheet("background-color: #EBEEF5; color: #909399; border: 1px dashed #DCDFE6; border-radius: 8px;")
        
        # 增加 stretch系数(1) 强制左右各占 50% 空间，防止只有单侧有图片时产生挤压
        image_layout.addWidget(self.ori_label, 1)
        image_layout.addWidget(self.res_label, 1)
        center_vbox.addLayout(image_layout)
        
        self.preview_btn = QPushButton(" 刷新效果预览 ")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #E6A23C; 
                color: white; 
                border: none;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #ebb563; }
            QPushButton:pressed { background-color: #cf9236; }
        """)
        self.preview_btn.clicked.connect(self.refresh_preview)
        center_vbox.addWidget(self.preview_btn)

        center_panel.setLayout(center_vbox)
        splitter.addWidget(center_panel)

        # 加载演示图片
        self.example_dir = os.path.join(bundle_dir, "example_images")
        self.example_img_path = os.path.join(self.example_dir, "picture.jpg")
        
        if os.path.exists(self.example_img_path):
            pixmap = QPixmap(self.example_img_path)
            # 放大最大尺寸到 800x800 以保证居中大图的清晰度和大小
            self.ori_label.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ori_label.setText("")

        # 2.3 右侧：流水线构建
        right_panel = QGroupBox("功能流水线")
        right_vbox = QVBoxLayout()
        
        top_btn_layout = QHBoxLayout()
        remove_btn = QPushButton("移除选中")
        remove_btn.setStyleSheet("color: #F56C6C; border-color: #FBC4C4; background-color: #FEF0F0;")
        remove_btn.clicked.connect(self.remove_step)
        clear_btn = QPushButton("清空全部")
        clear_btn.setStyleSheet("color: #909399; border-color: #DCDFE6; background-color: #F4F4F5;")
        clear_btn.clicked.connect(lambda: self.list_widget.clear())
        
        top_btn_layout.addWidget(remove_btn)
        top_btn_layout.addWidget(clear_btn)
        right_vbox.addLayout(top_btn_layout)

        self.list_widget = PipelineListWidget()
        self.list_widget.currentItemChanged.connect(self.on_item_selected)
        right_vbox.addWidget(QLabel("组件插槽:"))
        right_vbox.addWidget(self.list_widget)
        
        self.run_btn = QPushButton("运行流水线")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #67C23A; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #85ce61; }
            QPushButton:pressed { background-color: #5daf34; }
            QPushButton:disabled { background-color: #b3e19d; }
        """)
        self.run_btn.clicked.connect(self.run_pipeline)
        right_vbox.addWidget(self.run_btn)
        
        right_panel.setLayout(right_vbox)
        splitter.addWidget(right_panel)

        # 设置分离器的初始比例（左:中:右 = 2:6:2 => 中间占比更高）
        splitter.setSizes([200, 800, 200])

        # 添加全局进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

    def create_path_row(self, parent_layout, label_text):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(80)
        edit = QLineEdit()
        btn = QPushButton("浏览...")
        row.addWidget(label)
        row.addWidget(edit)
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return edit, btn

    def browse_folder(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if dir_path:
            line_edit.setText(dir_path)

    def remove_step(self):
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.list_widget.takeItem(current_row)

    def on_item_selected(self, current, previous):
        # 清空当前显示的参数
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
        if not current:
            return
            
        op_name = current.text()
        kwargs = current.data(Qt.UserRole)
        
        # 动态生成表单
        for key, value in kwargs.items():
            edit = QLineEdit(str(value))
            # 绑定文本修改事件，实时更新 item 的数据
            edit.textChanged.connect(lambda text, k=key, i=current: self.update_item_data(i, k, text))
            self.param_layout.addRow(QLabel(key + ":"), edit)

    def update_item_data(self, item, key, text):
        data = item.data(Qt.UserRole)
        # 根据原始数据类型尝试转换
        orig_val = data[key]
        try:
            if isinstance(orig_val, int):
                data[key] = int(text)
            elif isinstance(orig_val, float):
                data[key] = float(text)
            else:
                data[key] = text
            item.setData(Qt.UserRole, data)
        except ValueError:
            pass # 用户输入过程中的暂时不合法值被忽略
            
    def refresh_preview(self):
        if self.list_widget.count() == 0:
            QMessageBox.warning(self, "提示", "请先在流水线中添加至少一个模块以生成预览")
            return
            
        temp_dir = tempfile.mkdtemp()
        try:
            current_input = os.path.join(temp_dir, "input")
            os.makedirs(current_input)
            
            if not os.path.exists(self.example_img_path):
                QMessageBox.warning(self, "警告", f"未找到演示原图：{self.example_img_path}")
                return
            shutil.copy(self.example_img_path, current_input)
            
            # 使用 example_images 文件夹作为备用背景，防止背景合成失败
            old_bg = self.bg_edit.text()
            if not old_bg or not os.path.exists(old_bg):
                self.bg_edit.setText(self.example_dir)
                
            steps_count = self.list_widget.count()
            self.progress_bar.setMaximum(steps_count)
            self.progress_bar.setValue(0)
                
            for i in range(steps_count):
                item = self.list_widget.item(i)
                op_name = item.text()
                kwargs = item.data(Qt.UserRole)
                
                step_out = os.path.join(temp_dir, f"step{i}")
                if not os.path.exists(step_out):
                    os.makedirs(step_out)
                
                run_op(op_name, kwargs, current_input, step_out, self.bg_edit.text())
                current_input = step_out
                
                # 刷新进度条
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
                
            self.bg_edit.setText(old_bg) # 恢复原来的背景路径
            
            # 找到输出文件夹中的任意第一张图片并展示
            out_files = [f for f in os.listdir(current_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if out_files:
                out_img_path = os.path.join(current_input, out_files[0])
                pixmap = QPixmap(out_img_path)
                # 同样的 800x800 放缩设置
                self.res_label.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.res_label.setText("")
            else:
                self.res_label.setText("操作未生成有效图片图像")
                
        except Exception as e:
            QMessageBox.critical(self, "预览失败", f"预览过程发生错误:\n{str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run_pipeline(self):
        input_dir = self.input_edit.text()
        output_dir = self.output_edit.text()
        bg_dir = self.bg_edit.text()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "错误", "请同时指定输入和输出文件夹！")
            return
            
        if self.list_widget.count() == 0:
            QMessageBox.warning(self, "提示", "请先在流水线中添加至少一个操作")
            return

        # 禁用UI以防重复点击
        self.run_btn.setEnabled(False)
        self.run_btn.setText("正在执行中...")
        
        steps_count = self.list_widget.count()
        self.progress_bar.setRange(0, 0) # 设为左右来回滚动的动画（由于底层是根据文件夹整体处理，无法精准计算单张图片进度）
        
        # 抓取操作数据传递给线程
        steps = []
        for i in range(steps_count):
            item = self.list_widget.item(i)
            steps.append({
                "op_name": item.text(),
                "kwargs": copy.deepcopy(item.data(Qt.UserRole))
            })
            
        self.worker = PipelineWorker(steps, input_dir, output_dir, bg_dir)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.error_signal.connect(self.on_pipeline_error)
        self.worker.start()

    def update_progress(self, step_idx, op_name):
        self.run_btn.setText(f"执行中... 已完成: {op_name}")

    def on_pipeline_finished(self, final_output):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("运行流水线")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        QMessageBox.information(self, "成功", f"流水线处理完成！\n最终结果见:\n{final_output}")

    def on_pipeline_error(self, err_msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("运行流水线")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "执行中断", f"发生错误:\n{err_msg}")

MODERN_QSS = """
QMainWindow, QWidget {
    background-color: #F3F5F8;
    color: #333333;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE6;
    border-radius: 8px;
    margin-top: 15px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    top: 0px;
    color: #409EFF;
    font-weight: bold;
    background-color: transparent;
}
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE6;
    border-radius: 4px;
    padding: 6px 12px;
    color: #606266;
}
QPushButton:hover {
    color: #409EFF;
    border-color: #C6E2FF;
    background-color: #ECF5FF;
}
QPushButton:pressed {
    color: #3A8EE6;
    border-color: #3A8EE6;
    background-color: #FFFFFF;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE6;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #409EFF;
}
QLineEdit:focus {
    border-color: #409EFF;
}
QProgressBar {
    border: 1px solid #E4E7ED;
    border-radius: 5px;
    text-align: center;
    background-color: #EBEEF5;
    color: #606266;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #67C23A;
    border-radius: 4px;
}
QSplitter::handle {
    background-color: #E4E7ED;
    width: 6px;
    border-radius: 3px;
}
QSplitter::handle:hover {
    background-color: #C0C4CC;
}
"""

if __name__ == "__main__":
    from PyQt5.QtCore import QTimer
    app = QApplication(sys.path)
    app.setStyleSheet(MODERN_QSS)
    window = AugmentationApp()
    window.show()
    # 使用定时器延迟 0 毫秒调用全屏，确保 OS 窗口管理器彻底分发好普通窗口体后再进入最大化，解决“假全屏” Bug
    QTimer.singleShot(0, window.showMaximized)
    sys.exit(app.exec_())
