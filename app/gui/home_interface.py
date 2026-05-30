#home_interface.py
import os
import sys
import traceback
from io import StringIO
from typing import List, Dict

import cv2
import matplotlib
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use('QtAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] # 不然不支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

original_stdout = sys.stdout
sys.stdout = StringIO()
from qfluentwidgets import PushButton, TeachingTip, InfoBarIcon, TeachingTipTailPosition, \
    InfoBar, InfoBarPosition, PrimaryPushButton, IndeterminateProgressBar, ComboBox, ToolButton, FluentIcon

sys.stdout = original_stdout

from .elements.scroll_image import ScrollImage
from .elements.camera_capture import CameraDialog


class PredictingWorker(QThread):
    finished = pyqtSignal(dict, object)
    error = pyqtSignal(str)
    warning = pyqtSignal(str)

    _face_cascade = None

    def __init__(self, func, model_instance, img_array):
        super().__init__()
        self.func = func
        self.model_instance = model_instance
        self.img_array = img_array

    @classmethod
    def get_face_cascade(cls):
        if cls._face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' #type:ignore
            except AttributeError:
                import os
                cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')

            cls._face_cascade = cv2.CascadeClassifier(cascade_path)
            if cls._face_cascade.empty():
                cls._face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        return cls._face_cascade

    def detect_faces_opencv(self, img_array):
        face_cascade = self.get_face_cascade()

        if face_cascade.empty():
            return None, None

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

        if len(faces) == 0:
            return None, None

        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        x, y, w, h = faces[0]
        original_bbox = (x, y, w, h)  # 原始边界框

        margin = int(0.1 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_array.shape[1] - x, w + 2 * margin)
        h = min(img_array.shape[0] - y, h + 2 * margin)

        face_roi = img_array[y:y + h, x:x + w]

        return face_roi, original_bbox  # 返回人脸区域和原始边界框

    def run(self):
        try:
            # 检测人脸
            face_roi = None
            face_bbox = None  # 人脸边界框
            try:
                face_roi, face_bbox = self.detect_faces_opencv(self.img_array)
            except cv2.error as e:
                msg = f"{str(e)}\n{traceback.format_exc()}"
                self.error.emit(msg)

            if face_roi is None or face_roi.size == 0:
                # 未检测到人脸或检测失败，使用整张图片
                self.warning.emit("未检测到人脸")
                img_to_predict = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
                face_bbox = None
            else:
                # 检测到人脸
                img_to_predict = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            result = self.func(
                self.model_instance,
                img_to_predict
            )
            self.finished.emit(result, face_bbox)  # 返回结果和人脸位置

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
        # self.result_hBoxLayout.addWidget(self.result_label)

        self.result_vBoxLayout = QVBoxLayout()

        self.confidence_score_figure = Figure(figsize=(4, 4), dpi=100)
        self.confidence_score_canvas = FigureCanvasQTAgg(self.confidence_score_figure)

        self.result_vBoxLayout.addWidget(self.confidence_score_canvas)
        self.result_vBoxLayout.addWidget(self.result_label)

        self.result_hBoxLayout.addLayout(self.result_vBoxLayout)

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
        self.cam_btn = ToolButton(FluentIcon.CAMERA,self)
        self.predict_btn = PrimaryPushButton(text='开始识别')
        self.clear_btn = PushButton(text='Clear')

        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.cam_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        self.vBoxLayout.addLayout(btn_layout)

        self.open_btn.clicked.connect(self.open_image)
        self.cam_btn.clicked.connect(self.camera_capture)
        self.predict_btn.clicked.connect(self.predict)
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

    def plot_pie_chart(self, result: dict):
        """绘制饼图"""
        self.confidence_score_figure.clear()
        emotions = ('愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性')
        colors = ('#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#C7C7C7', '#999999')
        color_to_draw = [] # 这样可以保证颜色是一一对应的

        labels = []
        sizes = []
        other_size = 0

        for i, emotion in enumerate(emotions):
            if i in result:
                if result[i] > 0.01:  # 大于1%单独显示
                    labels.append(emotion)
                    color_to_draw.append(colors[i])
                    sizes.append(result[i])
                else:  # 小于等于1%合并到其他
                    other_size += result[i]

        if other_size > 0:
            labels.append('其他')
            sizes.append(other_size)

        if not sizes:
            self.confidence_score_canvas.hide()
            return

        # 创建子图
        ax = self.confidence_score_figure.add_subplot(111)

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda pct: f'{pct:.1f}%',
            colors=color_to_draw,
            startangle=90,
            textprops={'fontsize': 10}
        )

        # 设置标题
        ax.set_title('Confidence Score', fontsize=12, fontweight='bold', pad=20)

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        # 确保饼图是圆的
        ax.axis('equal')

        # 调整布局
        self.confidence_score_figure.tight_layout()

        self.confidence_score_canvas.show()
        self.confidence_score_canvas.draw()

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
            self.set_current_img(file_path)
        else:
            InfoBar.warning(
                title="WARNING",
                content="未选择图片",
                position=InfoBarPosition.TOP,
                parent=self
            )

    def camera_capture(self):
        window = CameraDialog(parent=self)
        window.captured.connect(self.set_current_img)
        window.show()

    def _camera_capture_error(self, msg: str):
        InfoBar.error(
            title="ERROR",
            content=msg,
            position=InfoBarPosition.TOP,
            parent=self
        )

    def set_current_img(self, file_path):
        self.result_label.setText("")
        self.confidence_score_figure.clear()
        if file_path:
            self.current_img = file_path
            image = QPixmap(file_path).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ) # 防止窗口被图片撑大

            self.result_label.setText("")
            self.image_label.setPixmap(image)
            self.confidence_score_figure.clear()

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

        img_array = cv2.imread(self.current_img)
        self.predicting_worker = PredictingWorker(
            func=self.model_op.predict_image,
            model_instance=self.current_model,
            img_array=img_array
        )
        self.predicting_worker.finished.connect(self._prediction_finished)
        self.predicting_worker.error.connect(self._prediction_error)
        self.predicting_worker.warning.connect(self._prediction_warning)
        self.predicting_worker.start()

    def _prediction_finished(self, result: Dict, face_bbox: tuple):
        self.in_progress_bar.stop()
        if result:
            InfoBar.success(
                title="SUCCESS",
                content="识别成功",
                position=InfoBarPosition.BOTTOM_RIGHT,
                parent=self,
                duration=1000
            )
            self.plot_pie_chart(result)
            emotions = ('愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性')
            key = max(result, key=result.get)
            self.result_label.setText(f"识别结果: {emotions[key]}({result[key]:.2%})")

            pixmap = QPixmap(self.current_img)

            # 如果检测到人脸，在 pixmap 上绘制框
            if face_bbox and not pixmap.isNull():
                from PyQt6.QtGui import QPainter, QPen, QFont, QColor

                new_pixmap = pixmap.copy() # 副本
                painter = QPainter(new_pixmap)
                pen = QPen(QColor(0, 255, 0))
                pen.setWidth(4)
                painter.setPen(pen)

                # 矩形框
                x, y, w, h = face_bbox
                painter.drawRect(x, y, w, h)

                font = QFont()
                font_size = 24
                font.setPointSize(font_size)
                painter.setFont(font)

                # painter.drawText(x, y - 5, "Face")
                painter.drawText(x, y + h + 4 + font_size, "Face")

                painter.end()
                pixmap = new_pixmap

            # 缩放显示
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)

        else:
            InfoBar.error(
                title="ERROR",
                content="识别失败",
                position=InfoBarPosition.TOP_RIGHT,
                parent=self
            )

    def _prediction_error(self, msg: str):
        self.in_progress_bar.stop()
        InfoBar.error(
            title="ERROR",
            content=f"识别失败:{msg}",
            position=InfoBarPosition.TOP_RIGHT,
            parent=self,
            duration=3000

        )

    def _prediction_warning(self, msg: str):
        self.in_progress_bar.stop()
        InfoBar.warning(
            title="WARNING",
            content=f"{msg}",
            position=InfoBarPosition.TOP_RIGHT,
            parent=self,
            duration=3000
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
        self.confidence_score_figure.clear()
        self.current_img = ""