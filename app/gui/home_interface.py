#home_interface.py
import logging
import os
import sys
import traceback
from io import StringIO
from typing import List, Dict, Any

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QFileDialog
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('QtAgg')

original_stdout = sys.stdout
sys.stdout = StringIO()
from qfluentwidgets import PushButton, TeachingTip, InfoBarIcon, TeachingTipTailPosition, \
    InfoBar, InfoBarPosition, PrimaryPushButton, IndeterminateProgressBar, ComboBox
sys.stdout = original_stdout

from .elements.scroll_image import ScrollImage

class PredictingWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, func, model_instance, img_array):
        super().__init__()
        self.func = func

        self.model_instance = model_instance
        self.img_array = img_array

    def run(self):
        try:
            result = self.func(
                self.model_instance,
                self.img_array
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

class SwitchingModelWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, func, model_name: str):
        super().__init__()
        self.func = func
        self.model_name = model_name

    def run(self):
        try:
            self.func(
                self.model_name,
            )
            self.finished.emit(self.model_name)
        except Exception as e:
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class HomeInterface(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.vBoxLayout = QVBoxLayout(self)

        self._entry_widget = parent

        #进度条
        self.in_progress_bar = IndeterminateProgressBar(self)
        self.in_progress_bar.stop()

        self.result_hBoxLayout = QHBoxLayout()
        # 图片显示
        self.image_label = ScrollImage()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 400px; border: none;")
        self.result_hBoxLayout.addWidget(self.image_label)
        # 结果显示
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px; color: black;")
        self.result_hBoxLayout.addWidget(self.result_label)

        self.result_vBoxLayout = QVBoxLayout()

        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)


        self.vBoxLayout.addLayout(self.result_vBoxLayout)
        self.vBoxLayout.addLayout(self.result_hBoxLayout)

        self.model_selection_comboBox = ComboBox(self)
        self.model_selection_comboBox.setPlaceholderText("Select a model.")

        items: List = list(getattr(self._entry_widget, "model_configs", {}).keys())
        self.model_selection_comboBox.addItems(items)
        self.model_selection_comboBox.setCurrentIndex(-1)
        self.model_selection_comboBox.currentTextChanged.connect(self.switch_model)
        self.vBoxLayout.addWidget(self.model_selection_comboBox)

        btn_layout = QHBoxLayout()

        self.open_btn = PushButton(text='打开图片')
        self.predict_btn = PrimaryPushButton(text='开始识别')
        self.clear_btn = PushButton(text='Clear')

        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        self.vBoxLayout.addLayout(btn_layout)

        self.open_btn.clicked.connect(self.open_image)
        def _start_predict():
            try:
                self.in_progress_bar.start()
                self.predict()
            finally:
                self.in_progress_bar.stop()
        self.predict_btn.clicked.connect(_start_predict)
        self.clear_btn.clicked.connect(self.clear)

        # 必须给子界面设置全局唯一的对象名
        self.setObjectName(text.replace(' ', '-'))

        self.current_img = ""
        self.predicting_worker = None
        self.switching_model_worker = None

    @property
    def current_model(self):
        return getattr(self._entry_widget, 'current_model', None)

    @property
    def model_op(self):
        return getattr(self._entry_widget, 'model_op', None)

    @property
    def logger(self):
        return getattr(self._entry_widget, "logger", None)

    def switch_model(self, model_name: str):

        if switch_model_func := getattr(self._entry_widget, 'switch_predicting_model', None):
            # switch_model(model_name)

            self.switching_model_worker = SwitchingModelWorker(switch_model_func, model_name)
            self.switching_model_worker.finished.connect(self._switching_model_finished)
            self.switching_model_worker.error.connect(self._switching_model_error)
            self.switching_model_worker.start()

    def _switching_model_finished(self, msg: str):
        InfoBar.success(
            title="INFO",
            content=f"Switching model to '{msg}'",
            position=InfoBarPosition.TOP,
            parent=self
        )

    def _switching_model_error(self, msg: str):
        InfoBar.error(
            title="ERROR",
            content=f"模型切换失败: {msg}",
            position=InfoBarPosition.TOP,
            parent=self
        )

    def open_image(self):
        file_picker = QFileDialog()
        file_path, _ = file_picker.getOpenFileName(filter="Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        self.result_label.setText("")
        if file_path:
            self.current_img = file_path
            image = QPixmap(file_path).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ) # 防止窗口被图片撑大
            self.image_label.setPixmap(image)
        else:
            InfoBar.warning(
                title="WARNING",
                content="未选择图片",
                position=InfoBarPosition.TOP,
                parent=self

            )

    def predict(self):
        self.in_progress_bar.start()
        if not self.current_model or not self.model_op:
            TeachingTip.create(
                target=self.model_selection_comboBox,
                icon=InfoBarIcon.ERROR,
                title='ERROR',
                content="模型未加载，请先加载模型",
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=2000,
                parent=self
            )
            self.logger.info("模型未加载，请先加载模型")
            return

        if not self.current_img:
            TeachingTip.create(
                target=self.predict_btn,
                icon=InfoBarIcon.WARNING,
                title='WARNING',
                content="请选择图片",
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=2000,
                parent=self
            )
            return

        if not self.current_img or not os.path.exists(self.current_img):
            TeachingTip.create(
                target=self.predict_btn,
                icon=InfoBarIcon.WARNING,
                title='WARNING',
                content="无效的图片路径",
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=2000,
                parent=self
            )
            self.logger.info("无效的图片路径")
            return

        try:
            # 转换为 ndarray
            img_array = cv2.imread(self.current_img)

            # 检查图片是否读取成功
            if img_array is None:
                self.result_label.setText("图片读取失败，请检查文件格式")
                return

            # 确保图片不为空
            if img_array.size == 0:
                self.result_label.setText("图片数据为空")
                return

            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # 限制图片最大尺寸
            max_size = 1024
            height, width = img_array.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_array = cv2.resize(img_array, (new_width, new_height))

            self.logger.info("Strat predicting")

            self.predicting_worker = PredictingWorker(func=self.model_op.predict_image, model_instance=self.current_model, img_array=img_array)
            self.predicting_worker.finished.connect(self._prediction_finished)
            self.predicting_worker.error.connect(self._prediction_error)
            self.predicting_worker.start()

            # result = model_op.predict_image(
            #     current_model,
            #     img_array
            # )

        except cv2.error as e:
            msg = f"OpenCV 错误: {e}"
            self.logger.error(msg)
            InfoBar.error(
                title="ERROR",
                content=msg,
                position=InfoBarPosition.TOP_RIGHT,
                parent = self
            )
        except Exception as e:
            msg = f"预测错误: {e}"
            self.logger.error(msg)
            InfoBar.error(
                title="ERROR",
                content=msg,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self
            )

    def _prediction_finished(self, result: Dict):
        if result:
            InfoBar.info(
                title="SUCCESS",
                content="识别成功",
                position=InfoBarPosition.TOP,
                parent=self

            )
            emotions = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性']
            key = max(result, key=result.get)
            self.result_label.setText(f"识别结果: {emotions[key]}({result[key]:.2%})")
        else:
            InfoBar.error(
                title="ERROR",
                content="识别失败",
                position=InfoBarPosition.TOP_RIGHT,
                parent=self

            )

    def _prediction_error(self, msg: str):

        InfoBar.error(
            title="ERROR",
            content=f"识别失败:{msg}",
            position=InfoBarPosition.TOP_RIGHT,
            parent=self

        )

    def clear(self):
        InfoBar.info(
            title="SUCCESS",
            content="已清空",
            position=InfoBarPosition.TOP_RIGHT,
            parent=self

        )
        self.result_label.setText("")
        self.image_label.setPixmap(QPixmap())
        self.current_img = ""