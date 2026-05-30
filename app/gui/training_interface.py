#training_interface.py
import traceback
from pathlib import Path
from typing import Dict

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout
from qfluentwidgets import InfoBarPosition, InfoBar

from .elements.model_info_card import ModelInfo, ModelStatus


class TrainWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, func, model_class, model_path, epochs, force):
        super().__init__()
        self.func = func
        self.model_class = model_class
        self.model_path = model_path
        self.epochs = epochs
        self.force = force

    def progress_callback(self, progress: int):
        self.progress.emit(progress)

    def run(self):
        try:
            self.func(
                self.model_class,
                self.model_path,
                self.epochs,
                self.force,
                self.progress_callback
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"训练失败: {str(e)}\n{traceback.format_exc()}")

class TrainingInterface(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)

        self._entry_widget = parent

        models: Dict = getattr(self._entry_widget, "model_configs", {})

        self.model_card_list = []
        for name in list(models.keys()):
            model_data = models[name]
            model_card = ModelInfo(model_name=name, model_data=model_data, parent=self)
            model_card.model_data = model_data
            self.hBoxLayout.addWidget(model_card)
            def create_train_handler(model_card):
                def handler():
                    self.train_model(model_card)

                return handler
            model_card.trainButton.clicked.connect(create_train_handler(model_card))

        self.setObjectName(text.replace(' ', '-'))

    @property
    def current_model(self):
        return getattr(self._entry_widget, 'current_model', None)

    @property
    def model_op(self):
        return getattr(self._entry_widget, 'model_op', None)

    @property
    def logger(self):
        return getattr(self._entry_widget, "logger", None)

    def train_model(self, model_card_widget: ModelInfo):
        if Path(model_card_widget.model_data["path"]).exists() and not model_card_widget.get_force():
            InfoBar.warning(
                title="Warning",
                content=f"模型文件{model_card_widget.model_data["path"]}已经存在",
                position=InfoBarPosition.TOP,
                parent=self
            )
            return

        model_card_widget.set_status(ModelStatus.PROCESSING)

        train_worker = TrainWorker(
            self.model_op.train_model,
            model_card_widget.model_data["class"],
            model_card_widget.model_data["path"],
            model_card_widget.get_epochs(),
            model_card_widget.get_force()
        )
        train_worker.progress.connect(model_card_widget.process_bar.setValue)

        model_card_widget.worker = train_worker

        model_card_widget.worker.finished.connect(
            lambda: self.on_train_finished(model_card_widget)
        )
        model_card_widget.worker.error.connect(
            lambda err: self.on_train_error(model_card_widget, err)
        )

        model_card_widget.worker.start()

    def on_train_finished(self, model_card_widget: ModelInfo):
        InfoBar.success(
            title="SUCCESS",
            content="训练完成",
            position=InfoBarPosition.TOP,
            parent=self

        )
        model_card_widget.set_status(status=ModelStatus.OK)
        self.logger.info(f"模型 {model_card_widget.model_data['class'].NAME} 训练完成")

    def on_train_error(self, model_card_widget: ModelInfo, error_msg: str):
        InfoBar.error(
            title="ERROR",
            content="训练失败",
            position=InfoBarPosition.TOP,
            parent=self

        )
        model_card_widget.set_status(status=ModelStatus.ERROR)
        self.logger.error(error_msg)

