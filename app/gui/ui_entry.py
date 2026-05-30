import logging
import sys
from io import StringIO

from PyQt6.QtCore import QSize, pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication
)

from app.gui.home_interface import HomeInterface
from app.gui.training_interface import TrainingInterface

original_stdout = sys.stdout
sys.stdout = StringIO()
from qfluentwidgets import FluentWindow, SplashScreen
from qfluentwidgets import FluentIcon as FIF
sys.stdout = original_stdout


class ModelLoaderThread(QThread):
    finished = pyqtSignal(object, dict)  # model_op, model_configs

    def run(self):
        from app.model.model_operator import ModelOperator
        from app.model.model import EmotionCNNModel1, EmotionCNNModel2, EmotionCNNModel3

        model_op = ModelOperator()
        model_configs = {
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

        self.finished.emit(model_op, model_configs)

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

        icon = "./res/images/splash.jpg"
        self.setWindowTitle('情感识别')
        self.setWindowIcon(QIcon(icon))
        # self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen = SplashScreen(QIcon(icon), self)
        self.splashScreen.setIconSize(QSize(256, 256))

        screen_geometry = self.screen().availableGeometry()
        screen_center = screen_geometry.center()

        window_width = int(screen_geometry.width() / 2 + 100)
        window_height = int(screen_geometry.height() / 3 * 2)

        window_x = screen_center.x() - int(window_width / 2 - 50)
        window_y = screen_center.y() - int(window_height / 2)

        self.setGeometry(window_x, window_y, window_width, window_height)

        self.show()
        self.splashScreen.show()

        self.model_op = None
        self.model_configs = {}
        # self.current_model = self.model_op.load_model(model_class=EmotionCNNModel1, model_path="./model_data/EmotionCNNModel1.pth")
        self.current_model = None

        # self.init_model() 这里换用多线程,不然不显示logo

        self.model_loader = ModelLoaderThread()
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.start()

        self.home_interface = None
        self.training_interface = None

    def on_model_loaded(self, model_op, model_configs):
        self.splashScreen.finish()
        self.model_op = model_op
        self.model_configs = model_configs

        self.home_interface = HomeInterface(parent=self, text="Home")
        self.addSubInterface(self.home_interface, icon=FIF.HOME,text="Home")
        self.training_interface = TrainingInterface(parent=self, text="Train")
        self.addSubInterface(self.training_interface, icon=FIF.DEVELOPER_TOOLS,text="Train")

    def init_model(self):
        from app.model.model_operator import ModelOperator
        from app.model.model import EmotionCNNModel1, EmotionCNNModel2, EmotionCNNModel3

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


    def switch_predicting_model(self, model_name: str):

        self.current_model = self.model_op.load_model(model_class=self.model_configs[model_name]["class"],
                                                      model_path=self.model_configs[model_name]["path"])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectorUI()
    window.show()
    sys.exit(app.exec())