# pyqt6_ui.py
import logging
import sys
from io import StringIO

from PyQt6.QtWidgets import (
    QApplication
)

from app.gui.home_interface import HomeInterface

original_stdout = sys.stdout
sys.stdout = StringIO()
from qfluentwidgets import FluentWindow
from qfluentwidgets import FluentIcon as FIF

sys.stdout = original_stdout

from app.model.model_operator import ModelOperator
from app.model.model import EmotionCNNModel1




class EmotionDetectorUI(FluentWindow):
    def __init__(self):
        super().__init__()

        self.logger = logging.Logger(name="EmotionDetectorUI")

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.setWindowTitle('情感识别')

        screen = self.screen().availableGeometry().center()
        self.setGeometry(screen.x() - int(self.width() / 2), screen.y() - int(self.height() / 2), self.width(),
                           self.height())

        self.model_op = ModelOperator()
        self.current_model = self.model_op.load_model(model_class=EmotionCNNModel1, model_path="./model_data/EmotionCNNModel1.pth")

        self.home_interface = HomeInterface(parent=self, text="Home")
        self.addSubInterface(self.home_interface, icon=FIF.HOME,text="Home")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectorUI()
    window.show()
    sys.exit(app.exec())