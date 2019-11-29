#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 25
# #######################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtWidgets, QtGui
from MorphingGUI import *
from Morphing import *
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
        self.leftImagePath = None
        self.rightImagePath = None
        self.leftPointsPath = None
        self.rightPointsPath = None

        # slider init
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))
        self.Slider.setMaximum(21)
        self.Slider.sliderMoved.connect(self.sliderValue)

        # load button init
        self.btnLoadLeft.clicked.connect(self.loadLeft)
        self.btnLoadRight.clicked.connect(self.loadRight)

        # loaded state
        self.checkBox.toggled.connect(self.showTri)

    def showTri(self):
        if not self.checkBox.isChecked():
            return
        (leftTri, rightTri) = loadTriangles(self.leftPointsPath, self.rightPointsPath)
        for each in leftTri:
            points = getPoints(each.vertices)
            for eachPoint in points:
                self.imageLeftViewer.scene.addItem(eachPoint)
        for each in rightTri:
            points = getPoints(each.vertices)
            for eachPoint in points:
                self.imageRightViewer.scene.addItem(eachPoint)

    def loadLeft(self):
        self.leftImagePath, _ = QFileDialog.getOpenFileName(self, caption='Open file ...', filter="files (*.png *.jpg)")
        if not self.leftImagePath:
            return
        image, self.leftPointsPath = loadImage(self.leftImagePath)
        self.leftLoaded = True
        self.imageLeftViewer.scene = QtWidgets.QGraphicsScene()
        self.imageLeftViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageLeftViewer.setScene(self.imageLeftViewer.scene)
        self.btnEnable()

    def loadRight(self):
        self.rightImagePath, _ = QFileDialog.getOpenFileName(self, caption='Open file ...', filter="files (*.png *.jpg)")
        if not self.rightImagePath:
            return
        image, self.rightPointsPath = loadImage(self.rightImagePath)
        self.rightLoaded = True
        self.imageRightViewer.scene = QtWidgets.QGraphicsScene()
        self.imageRightViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageRightViewer.setScene(self.imageRightViewer.scene)
        self.btnEnable()

    def sliderValue(self):
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))

    def btnEnable(self):
        if self.leftLoaded and self.rightLoaded:
            self.Slider.setDisabled(False)
            self.btnBlend.setDisabled(False)


def loadImage(filePath):
    # image = imageio.imread(filePath)
    pointsPath = filePath + '.txt'
    image = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(open(filePath, 'rb').read()))
    # image = image.scaledToHeight(200)
    # image = image.scaledToWidth(290)
    image = image.scaled(290, 200, QtCore.Qt.KeepAspectRatio)
    return image, pointsPath


def getPoints(vertices):
    point = [QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1]),
             QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1]),
             QtWidgets.QGraphicsLineItem(vertices[1][0], vertices[1][1], vertices[1][0], vertices[1][1])]
    for index, each in enumerate(point):
        each.setPen(QtGui.QPen(QtCore.Qt.red, 1))
    return point

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()
