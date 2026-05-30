import os
from datetime import datetime

import cv2
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
from qfluentwidgets import PrimaryPushButton, PushButton


class CameraDialog(QDialog):
    captured = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("摄像头")
        self.setModal(True)
        self.resize(640, 520)

        self.cap = None
        self.timer = QTimer()
        self.current_frame = None

        # 尝试打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error.emit("无法打开摄像头，请检查设备是否连接")
            self.close_btn = PushButton(text="close")
            self.close_btn.clicked.connect(self.close)
            layout = QVBoxLayout(self)
            error_label = QLabel("摄像头未找到或无法访问")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
            layout.addWidget(self.close_btn)
            return

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        layout = QVBoxLayout(self)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        self.capture_btn = PrimaryPushButton(text="capture")
        self.close_btn = PushButton(text="close")
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

        self.capture_btn.clicked.connect(self.capture)
        self.close_btn.clicked.connect(self.close)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))
            self.current_frame = frame

    def capture(self) -> str:
        os.makedirs("./temp", exist_ok=True)
        name = f"./temp/camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(name, self.current_frame)
        self.close()
        self.captured.emit(name)
        return name

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()