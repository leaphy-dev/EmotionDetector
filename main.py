import sys
import torch # type: ignore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.gui.ui_entry import EmotionDetectorUI


if __name__ == '__main__':
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)  # 启用高 DPI 缩放
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)  # 使用高 DPI 图标
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)
    app = QApplication(sys.argv)
    window = EmotionDetectorUI()
    window.show()
    sys.exit(app.exec())