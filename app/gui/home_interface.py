import logging
import os

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QFileDialog
from qfluentwidgets import SubtitleLabel, setFont, PushButton, TeachingTip, InfoBarIcon, TeachingTipTailPosition, \
    InfoBar, InfoBarPosition, PrimaryPushButton

from .elements.scroll_image import ScrollImage

class HomeInterface(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.vBoxLayout = QVBoxLayout(self)

        self._entry_widget = parent

        self.logger = getattr(parent, "logger")

        # 图片显示
        self.image_label = ScrollImage()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 400px;")
        self.vBoxLayout.addWidget(self.image_label)

        # 结果显示
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        self.vBoxLayout.addWidget(self.result_label)
        btn_layout = QHBoxLayout()

        self.open_btn = PushButton('打开图片')
        self.predict_btn = PrimaryPushButton('开始识别')
        self.clear_btn = PushButton('Clear')

        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        self.vBoxLayout.addLayout(btn_layout)

        self.open_btn.clicked.connect(self.open_image)
        self.predict_btn.clicked.connect(self.predict)
        self.clear_btn.clicked.connect(self.clear)

        # 必须给子界面设置全局唯一的对象名
        self.setObjectName(text.replace(' ', '-'))

        self.current_img = ""

    def open_image(self):
        file_picker = QFileDialog()
        file_path, _ = file_picker.getOpenFileName(filter="Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.current_img = file_path
            image = QPixmap(file_path).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ) # 防止窗口被图片撑大
            self.image_label.setPixmap(image)

    def predict(self):
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

        if not hasattr(self._entry_widget, 'current_model') or not hasattr(self._entry_widget, 'model_op'):
            TeachingTip.create(
                target=self.predict_btn,
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

            model_op = getattr(self._entry_widget, "model_op")
            current_model = getattr(self._entry_widget, "current_model")

            result = model_op.predict_image(
                current_model,
                img_array
            )

            if result:
                InfoBar.info(
                    title="SUCCESS",
                    content="识别成功",
                    position=InfoBarPosition.TOP,
                    parent = self

                )
                self.result_label.setText(f"识别结果: {result}")
            else:
                InfoBar.error(
                    title="ERROR",
                    content="识别失败",
                    position=InfoBarPosition.TOP_RIGHT,
                    parent = self

                )
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
    def clear(self):
        self.image_label.setPixmap(QPixmap())
        self.current_img = ""