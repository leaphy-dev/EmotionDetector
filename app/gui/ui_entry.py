import logging
import sys
from io import StringIO

from PyQt6.QtWidgets import (
    QApplication
)

from app.gui.home_interface import HomeInterface
from app.gui.training_interface import TrainingInterface

original_stdout = sys.stdout
sys.stdout = StringIO()
from qfluentwidgets import FluentWindow
from qfluentwidgets import FluentIcon as FIF

sys.stdout = original_stdout

from app.model.model_operator import ModelOperator
from app.model.model import EmotionCNNModel1, EmotionCNNModel2, EmotionCNNModel3




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

        self.model_configs = {
            EmotionCNNModel1.NAME: {
                "class": EmotionCNNModel1,
                "path": "./model_data/EmotionCNNModel1.pth",
                "description": EmotionCNNModel1.__doc__
            },
            EmotionCNNModel2.NAME: {
                "class": EmotionCNNModel2,
                "path": "./model_data/EmotionCNNModel2.pth",
                "description": EmotionCNNModel2.__doc__
            },
            EmotionCNNModel3.NAME: {
                "class": EmotionCNNModel3,
                "path": "./model_data/EmotionCNNModel3.pth",
                "description": EmotionCNNModel3.__doc__
            }
        }


        # self.current_model = self.model_op.load_model(model_class=EmotionCNNModel1, model_path="./model_data/EmotionCNNModel1.pth")
        self.current_model = None

        self.home_interface = HomeInterface(parent=self, text="Home")
        self.addSubInterface(self.home_interface, icon=FIF.HOME,text="Home")
        self.training_interface = TrainingInterface(parent=self, text="Train")
        self.addSubInterface(self.training_interface, icon=FIF.DEVELOPER_TOOLS,text="Train")

    def switch_predicting_model(self, model_name: str):

        self.current_model = self.model_op.load_model(model_class=self.model_configs[model_name]["class"],
                                                      model_path=self.model_configs[model_name]["path"])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectorUI()
    window.show()
    sys.exit(app.exec())