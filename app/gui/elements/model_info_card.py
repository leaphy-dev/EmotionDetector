from enum import Enum

from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout
from qfluentwidgets import BodyLabel, IconWidget, InfoBarIcon, HeaderCardWidget, PrimaryPushButton, \
    CheckBox, IndeterminateProgressBar, SpinBox


class ModelStatus(int, Enum):
    OK = 0
    ERROR = 1
    PROCESSING = 2

class ModelInfo(HeaderCardWidget):
    """模型信息卡片"""

    def __init__(self, model_name: str, model_data, parent=None):
        super().__init__(parent)

        self.model_data = model_data
        self.model_status = 0

        self.setTitle(model_name)
        self.descriptionLabel = BodyLabel(self.model_data.get("description", ""), self)
        self.statusIcon = IconWidget(InfoBarIcon.SUCCESS, self)
        self.inProgressBar = IndeterminateProgressBar(self)
        self.inProgressBar.stop()
        self.statusLabel = BodyLabel('Ready', self)

        self.epochsLabel = BodyLabel('训练轮数:', self)
        self.epochsSpinBox = SpinBox(self)
        self.epochsSpinBox.setRange(1, 1000)
        self.epochsSpinBox.setValue(5)
        self.epochsSpinBox.setSuffix(' 轮')
        self.epochsSpinBox.setFixedWidth(100)

        # 训练按钮
        self.trainButton = PrimaryPushButton('开始训练', self)

        # force复选框
        self.forceCheckBox = CheckBox('强制训练 (覆盖已有模型)', self)
        self.forceCheckBox.setChecked(False)

        self.vBoxLayout = QVBoxLayout()
        self.statusLayout = QHBoxLayout()
        self.epochsLayout = QHBoxLayout()
        self.buttonLayout = QHBoxLayout()
        self.forceLayout = QHBoxLayout()

        self.statusIcon.setFixedSize(16, 16)

        self.statusLayout.setSpacing(10)
        self.epochsLayout.setSpacing(8)
        self.buttonLayout.setSpacing(10)
        self.forceLayout.setSpacing(8)
        self.vBoxLayout.setSpacing(12)

        self.statusLayout.setContentsMargins(0, 0, 0, 0)
        self.epochsLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.forceLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)

        self.statusLayout.addWidget(self.statusIcon)
        self.statusLayout.addWidget(self.statusLabel)
        self.statusLayout.addStretch()

        self.epochsLayout.addWidget(self.epochsLabel)
        self.epochsLayout.addWidget(self.epochsSpinBox)
        self.epochsLayout.addStretch()

        self.buttonLayout.addStretch()
        self.buttonLayout.addWidget(self.trainButton)

        self.forceLayout.addStretch()
        self.forceLayout.addWidget(self.forceCheckBox)
        self.forceLayout.addStretch()

        self.vBoxLayout.addWidget(self.descriptionLabel)
        self.vBoxLayout.addLayout(self.statusLayout)
        self.vBoxLayout.addWidget(self.inProgressBar)
        self.vBoxLayout.addLayout(self.epochsLayout)
        self.vBoxLayout.addLayout(self.buttonLayout)
        self.vBoxLayout.addLayout(self.forceLayout)  # 添加force复选框行

        self.viewLayout.addLayout(self.vBoxLayout)

        # 连接训练按钮信号
        # self.trainButton.clicked.connect(self._on_train_clicked)

    def set_train_button_enabled(self, enabled: bool):
        """设置训练按钮是否可用"""
        self.trainButton.setEnabled(enabled)

    def set_train_button_text(self, text: str):
        self.trainButton.setText(text)

    def set_train_button_visible(self, visible: bool):
        self.trainButton.setVisible(visible)

    def set_status(self, status: int = 0):
        if status == ModelStatus.OK:
            self.set_train_button_enabled(True)
            self.forceCheckBox.setEnabled(True)
            self.inProgressBar.stop()
            self.statusIcon.setIcon(InfoBarIcon.SUCCESS)
            self.statusLabel.setText("Ready")
            self.model_status = ModelStatus.OK

        elif status == ModelStatus.PROCESSING:
            self.inProgressBar.start()
            self.set_train_button_enabled(False)
            self.forceCheckBox.setEnabled(False)
            self.statusIcon.setIcon(InfoBarIcon.INFORMATION)
            self.statusLabel.setText("Training")
            self.model_status = ModelStatus.PROCESSING

        else:
            self.set_train_button_enabled(True)
            self.forceCheckBox.setEnabled(True)
            self.inProgressBar.stop()
            self.statusIcon.setIcon(InfoBarIcon.ERROR)
            self.statusLabel.setText("Error")
            self.model_status = ModelStatus.ERROR


    def set_model_description(self, description: str):
        self.descriptionLabel.setText(description)

    def get_train_button(self):
        return self.trainButton

    def get_epochs(self) -> int:
        return self.epochsSpinBox.value()

    def set_epochs(self, value: int):
        self.epochsSpinBox.setValue(value)

    def set_epochs_range(self, min_val: int, max_val: int):
        self.epochsSpinBox.setRange(min_val, max_val)

    def set_epochs_visible(self, visible: bool):
        self.epochsLabel.setVisible(visible)
        self.epochsSpinBox.setVisible(visible)

    def get_force(self) -> bool:
        return self.forceCheckBox.isChecked()

    def set_force(self, checked: bool):
        self.forceCheckBox.setChecked(checked)

    def set_force_visible(self, visible: bool):
        self.forceCheckBox.setVisible(visible)

    def set_force_text(self, text: str):
        self.forceCheckBox.setText(text)