#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 25
# #######################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtWidgets
from MorphingGUI import *
import numpy as np
import imageio


class MorphingApp(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MorphingApp, self).__init__(parent)
        self.setupUi(self)
        self.alphaShow.setDisabled(True)
        self.Slider.setDisabled(True)
        self.btnBlend.setDisabled(True)
        self.leftLoaded = False
        self.rightLoaded = False
        self.leftImage = None
        self.rightImage = None
        self.leftPointsPath = None
        self.rightPointsPath = None

        # slider init
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))
        self.Slider.setMaximum(21)
        self.Slider.sliderMoved.connect(self.sliderValue)

        # load button init
        self.btnLoadLeft.clicked.connect(self.loadLeft)
        self.btnLoadLeft.clicked.connect(self.loadRight)



    def loadLeft(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open PNG file ...', filter="PNG files (*.png)")
        if not filePath:
            return
        leftImage, self.leftPointsPath = loadImage(filePath)
        self.leftLoaded = True
        self.imageLeftViewer.scene = QtWidgets.QGraphicsScene()
        self.imageLeftViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(leftImage))
        self.imageLeftViewer.setScene(self.imageLeftViewer.scene)



    def loadRight(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open PNG file ...', filter="PNG files (*.png)")
        if not filePath:
            return
        self.rightImage, self.rightPointsPath = loadImage(filePath)
        self.rightLoaded = True

    def sliderValue(self):
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))


def loadImage(filePath):
    # image = imageio.imread(filePath)
    pointsPath = filePath[0:-4] + '.txt'
    image = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(open(filePath, 'rb').read()))
    return image, pointsPath

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()
