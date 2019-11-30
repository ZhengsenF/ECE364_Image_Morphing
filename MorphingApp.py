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

        # widget initialization
        self.alphaShow.setDisabled(True)
        self.Slider.setDisabled(True)
        self.btnBlend.setDisabled(True)
        self.checkBox.setDisabled(True)

        # variable initialization
        self.leftLoaded = False
        self.rightLoaded = False
        self.leftImagePath = None
        self.rightImagePath = None
        self.leftPointsPath = None
        self.rightPointsPath = None
        self.leftImageShape = None
        self.rightImageShape = None

        # points and indexes initialization
        self.leftPoints = []
        self.rightPoints = []
        self.leftFromFile = 0
        self.rightFromFile = 0

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
            self.imageLeftViewer.scene.clear()
            self.imageRightViewer.scene.clear()
            # reload left
            image, self.leftPointsPath = loadImage(self.leftImagePath)
            self.imageLeftViewer.scene = QtWidgets.QGraphicsScene()
            self.imageLeftViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
            self.imageLeftViewer.setScene(self.imageLeftViewer.scene)
            self.imageLeftViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageLeftViewer.scene),
                                           QtCore.Qt.KeepAspectRatio)
            # reload right
            image, self.rightPointsPath = loadImage(self.rightImagePath)
            self.imageRightViewer.scene = QtWidgets.QGraphicsScene()
            self.imageRightViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
            self.imageRightViewer.setScene(self.imageRightViewer.scene)
            self.imageRightViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageRightViewer.scene),
                                            QtCore.Qt.KeepAspectRatio)
            return
        (leftTriangles, rightTriangles) = loadTriangles(self.leftPointsPath, self.rightPointsPath)
        for each in leftTriangles:
            lines = getLines(each.vertices)
            for eachPoint in each.vertices:
                self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                      brush=QtGui.QBrush(QtCore.Qt.red))
            for eachLine in lines:
                self.imageLeftViewer.scene.addItem(eachLine)

        for each in rightTriangles:
            lines = getLines(each.vertices)
            for eachPoint in each.vertices:
                self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                       brush=QtGui.QBrush(QtCore.Qt.red))
            for eachLine in lines:
                self.imageRightViewer.scene.addItem(eachLine)

    # load starting image and its points file
    def loadLeft(self):
        # open image
        self.leftImagePath, _ = QFileDialog.getOpenFileName(self, caption='Open file ...', filter="files (*.png *.jpg)")
        if not self.leftImagePath:
            return
        image, self.leftPointsPath = loadImage(self.leftImagePath)
        self.leftLoaded = True
        self.leftImageShape = imageio.imread(self.leftImagePath).shape
        # manipulate viewer
        self.imageLeftViewer.scene = QtWidgets.QGraphicsScene()
        self.imageLeftViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageLeftViewer.setScene(self.imageLeftViewer.scene)
        self.imageLeftViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageLeftViewer.scene),
                                       QtCore.Qt.KeepAspectRatio)
        self.btnEnable()
        # open points
        self.leftPoints, self.leftFromFile = loadPoints(self.leftPointsPath)
        for eachPoint in self.leftPoints:
            self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.red))

    # load ending image and its points file
    def loadRight(self):
        # open image
        self.rightImagePath, _ = QFileDialog.getOpenFileName(self, caption='Open file ...', filter="files (*.png *.jpg)")
        if not self.rightImagePath:
            return
        image, self.rightPointsPath = loadImage(self.rightImagePath)
        self.rightLoaded = True
        self.rightImageShape = imageio.imread(self.rightImagePath).shape
        # manipulate viewer
        self.imageRightViewer.scene = QtWidgets.QGraphicsScene()
        self.imageRightViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageRightViewer.setScene(self.imageRightViewer.scene)
        self.imageRightViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageRightViewer.scene),
                                       QtCore.Qt.KeepAspectRatio)
        self.btnEnable()
        # open points
        self.rightPoints, self.rightFromFile = loadPoints(self.rightPointsPath)
        for eachPoint in self.rightPoints:
            self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.red))

    # text box on the right of slider bar
    def sliderValue(self):
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))

    # check if both images are loaded. If so, enable rest of the functionality of the app
    def btnEnable(self):
        if self.leftLoaded and self.rightLoaded:
            self.Slider.setDisabled(False)
            self.btnBlend.setDisabled(False)
            self.checkBox.setDisabled(False)


# load image in QPixmap form from image file path
# returns such image and the path of its points map
def loadImage(filePath):
    # image = imageio.imread(filePath)
    pointsPath = filePath + '.txt'
    image = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(open(filePath, 'rb').read()))
    # image = image.scaledToHeight(200)
    # image = image.scaledToWidth(290)
    # image = image.scaled(290, 200)
    return image, pointsPath


# take filePath of the points
# returns a list of points and points from file (ending index + 1)
def loadPoints(filePath):
    if not os.path.exists(filePath):
        return [], 0
    # read from file
    with open(filePath) as file:
        lines = file.readlines()
    points = []
    for eachLine in lines:
        data = eachLine.split()
        data = [float(data[0]), float(data[1])]
        points.append(data)
    return points, len(points)


# takes triangle's vertices and image size returns lines to be plotted
def getLines(vertices):
    lines = [QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1]),
             QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1]),
             QtWidgets.QGraphicsLineItem(vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1])]
    for index, each in enumerate(lines):
        each.setPen(QtGui.QPen(QtCore.Qt.red, 1))
    return lines


if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()
