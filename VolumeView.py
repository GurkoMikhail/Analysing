from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSizePolicy
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph import mkQApp
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem,  GLBoxItem


class VolumeView(QWidget):
    
    def __init__(self, parent=None, name="VolumeView", *args):
        super().__init__(parent, *args)
        self.viewBox = QHBoxLayout(self)
        self.openGLWidget = GLViewWidget(self)
        self.viewBox.addWidget(self.openGLWidget)
        self.graphicsView = GraphicsLayoutWidget(self)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setMaximumSize(QtCore.QSize(200, 16777215))
        self.viewBox.addWidget(self.graphicsView)


if __name__ == "__main__":
    from pyqtgraph import mkQApp
    from PyQt5.uic import loadUi

    mkQApp()
    mainWindow = loadUi("UI/volume_visualization.ui")
    mainWindow.show()
    QtGui.QApplication.exec()
