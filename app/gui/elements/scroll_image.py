from PyQt6.QtCore import QEasingCurve, Qt
from PyQt6.QtGui import QPixmap
from qfluentwidgets import SmoothScrollArea, PixmapLabel


class ScrollImage(SmoothScrollArea):

    def __init__(self, pixmap: QPixmap = None):
        super().__init__()
        if not pixmap:
            pixmap = QPixmap()
        self.label = PixmapLabel(self)
        self.label.setPixmap(pixmap)

        # display the handle only when mouse hover the scroll bar region
        # self.delegate.vScrollBar.setHandleDisplayMode(ScrollBarHandleDisplayMode.ON_HOVER)

        # customize scroll animation
        self.setScrollAnimation(Qt.Orientation.Vertical, 500, QEasingCurve(QEasingCurve.Type.OutQuint))
        self.setScrollAnimation(Qt.Orientation.Horizontal, 500, QEasingCurve(QEasingCurve.Type.OutQuint))

        self.horizontalScrollBar().setValue(1900)
        self.setWidget(self.label)
        self.resize(960, 640)


    def setPixmap(self, pixmap: QPixmap) -> None:
        self.label.setPixmap(pixmap)

