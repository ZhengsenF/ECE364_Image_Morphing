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
        self.leftNew = False
        self.rightNew = False
        self.correspondence = False
        self.leftImagePath = None
        self.rightImagePath = None
        self.leftPointsPath = None
        self.rightPointsPath = None
        self.leftImageShape = None
        self.rightImageShape = None
        self.leftPointFile = None
        self.rightPointFile = None
        self.leftTempPoint = None
        self.rightTempPoint = None

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

        # mouse click event
        self.imageLeftViewer.mousePressEvent = self.leftClicked
        self.imageRightViewer.mousePressEvent = self.rightClicked
        self.mousePressEvent = self.elseClicked

        # keyboard event
        self.keyPressEvent = self.back

        # blend
        self.btnBlend.clicked.connect(self.blend)

    def back(self, event):
        if event.key() == QtCore.Qt.Key_Backspace:
            if self.rightNew:
                self.rightNew = False
                self.rightReload()
            elif self.leftNew:
                self.leftNew = False
                self.leftReload()

    def blend(self):
        leftPoints = np.array(self.leftPoints)
        rightPoints = np.array(self.rightPoints)
        leftDelaunay = Delaunay(leftPoints)
        leftTriangle = triangleFromDelaunay(leftDelaunay, leftPoints)
        rightTriangle = triangleFromDelaunay(leftDelaunay, rightPoints)
        leftImage = imageio.imread(self.leftImagePath)
        rightImage = imageio.imread(self.rightImagePath)
        morpher = ColorMorpher(leftImage, leftTriangle, rightImage, rightTriangle)
        alpha = self.Slider.value() / 20
        morphed = morpher.getImageAtAlpha(alpha)
        imagePath = 'resultColor.png'
        imagePath = os.path.join(os.getcwd(), imagePath)
        imageio.imwrite(imagePath, morphed)
        image, _ = loadImage(imagePath)
        self.imageBlendViwer.scene = QtWidgets.QGraphicsScene()
        self.imageBlendViwer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageBlendViwer.setScene(self.imageBlendViwer.scene)
        self.imageBlendViwer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageBlendViwer.scene),
                                       QtCore.Qt.KeepAspectRatio)
        self.imageBlendViwer.update()

    def elseClicked(self, position):
        if not (self.leftNew and self.rightNew):
            return
        self.persistPoint()

    def leftClicked(self, position):
        # not in loaded state
        if not (self.rightLoaded and self.leftLoaded):
            return
        if self.leftNew and not self.rightNew:
            return
        if self.leftNew and self.rightNew:
            self.persistPoint()
        x = self.imageLeftViewer.mapToScene(position.pos()).x()
        y = self.imageLeftViewer.mapToScene(position.pos()).y()
        self.leftNew = True
        self.leftTempPoint = [round(x, 1), round(y, 1)]
        self.leftReload()

    def rightClicked(self, position):
        # not in loaded state
        if not (self.rightLoaded and self.leftLoaded):
            return
        if self.rightNew:
            return
        if not self.leftNew:
            return
        x = self.imageRightViewer.mapToScene(position.pos()).x()
        y = self.imageRightViewer.mapToScene(position.pos()).y()
        self.rightNew = True
        self.rightTempPoint = [round(x, 1), round(y, 1)]
        self.rightReload()

    # write points to file and reset others
    def persistPoint(self):
        self.leftPoints.append(self.leftTempPoint)
        self.rightPoints.append(self.rightTempPoint)
        self.leftPointFile.write('{:8.1f}{:8.1f}\n'.format(self.leftPoints[-1][0], self.leftPoints[-1][1]))
        self.rightPointFile.write('{:8.1f}{:8.1f}\n'.format(self.rightPoints[-1][0], self.rightPoints[-1][1]))
        fileSave(self.leftPointFile)
        fileSave(self.rightPointFile)
        self.leftNew = False
        self.rightNew = False
        self.leftReload()
        self.rightReload()

    # reload the image, points and lines
    def leftReload(self):
        if len(self.leftPoints) >= 3:
            self.checkBox.setDisabled(False)
        image, self.leftPointsPath = loadImage(self.leftImagePath)
        self.imageLeftViewer.scene = QtWidgets.QGraphicsScene()
        self.imageLeftViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageLeftViewer.setScene(self.imageLeftViewer.scene)
        self.imageLeftViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageLeftViewer.scene),
                                       QtCore.Qt.KeepAspectRatio)
        for eachPoint in self.leftPoints[0:self.leftFromFile]:
            self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.red))
        for eachPoint in self.leftPoints[self.leftFromFile:]:
            self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.blue))
        if self.leftNew:
            eachPoint = self.leftTempPoint
            self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.green))
        if self.checkBox.isChecked():
            leftDelaunay = Delaunay(self.leftPoints)
            leftTriangles = triangleFromDelaunay(leftDelaunay, self.leftPoints)
            # display triangles
            for each in leftTriangles:
                lines = getLines(each.vertices)
                for eachLine in lines:
                    self.imageLeftViewer.scene.addItem(eachLine)

    def rightReload(self):
        image, self.rightPointsPath = loadImage(self.rightImagePath)
        self.imageRightViewer.scene = QtWidgets.QGraphicsScene()
        self.imageRightViewer.scene.addItem(QtWidgets.QGraphicsPixmapItem(image))
        self.imageRightViewer.setScene(self.imageRightViewer.scene)
        self.imageRightViewer.fitInView(QtWidgets.QGraphicsScene.itemsBoundingRect(self.imageRightViewer.scene),
                                        QtCore.Qt.KeepAspectRatio)
        for eachPoint in self.rightPoints[0:self.rightFromFile]:
            self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                   brush=QtGui.QBrush(QtCore.Qt.red))
        for eachPoint in self.rightPoints[self.rightFromFile:]:
            self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                   brush=QtGui.QBrush(QtCore.Qt.blue))
        if self.rightNew:
            eachPoint = self.rightTempPoint
            self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                   brush=QtGui.QBrush(QtCore.Qt.green))
        if self.checkBox.isChecked():
            leftDelaunay = Delaunay(self.rightPoints)
            rightTriangles = triangleFromDelaunay(leftDelaunay, self.rightPoints)
            # display triangles
            for each in rightTriangles:
                lines = getLines(each.vertices)
                for eachLine in lines:
                    self.imageRightViewer.scene.addItem(eachLine)

    def showTri(self):
        # box uncheck
        if not self.checkBox.isChecked():
            self.leftReload()
            self.rightReload()
            return
        # box checked
        # load triangles
        leftDelaunay = Delaunay(self.leftPoints)
        leftTriangles = triangleFromDelaunay(leftDelaunay, self.leftPoints)
        rightTriangles = triangleFromDelaunay(leftDelaunay, self.rightPoints)
        # display triangles
        for each in leftTriangles:
            lines = getLines(each.vertices)
            for eachLine in lines:
                self.imageLeftViewer.scene.addItem(eachLine)

        for each in rightTriangles:
            lines = getLines(each.vertices)
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
        self.leftPoints, self.leftFromFile, self.correspondence = loadPoints(self.leftPointsPath)
        for eachPoint in self.leftPoints:
            self.imageLeftViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                  brush=QtGui.QBrush(QtCore.Qt.red))
        if self.leftLoaded and self.rightLoaded and not self.correspondence:
            self.leftPointsPath = 'tempLeft.txt'
            self.leftPointFile = open(self.leftPointsPath, 'w')
            self.rightPointsPath = 'tempRight.txt'
            self.rightPointFile = open(self.rightPointsPath, 'w')
        elif self.leftLoaded and self.rightLoaded and self.correspondence:
            self.leftPointFile = open(self.leftPointsPath, 'a')
            self.rightPointFile = open(self.rightPointsPath, 'a')
            self.leftPointFile.write('\n')
            self.rightPointFile.write('\n')

    # load ending image and its points file
    def loadRight(self):
        # open image
        self.rightImagePath, _ = QFileDialog.getOpenFileName(self, caption='Open file ...',
                                                             filter="files (*.png *.jpg)")
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
        self.rightPoints, self.rightFromFile, self.correspondence = loadPoints(self.rightPointsPath)
        for eachPoint in self.rightPoints:
            self.imageRightViewer.scene.addEllipse(QtCore.QRectF(eachPoint[0] - 10, eachPoint[1] - 10, 20, 20),
                                                   brush=QtGui.QBrush(QtCore.Qt.red))
        if self.leftLoaded and self.rightLoaded and not self.correspondence:
            self.leftPointsPath = 'tempLeft.txt'
            self.leftPointFile = open(self.leftPointsPath, 'w')
            self.rightPointsPath = 'tempRight.txt'
            self.rightPointFile = open(self.rightPointsPath, 'w')
        elif self.leftLoaded and self.rightLoaded and self.correspondence:
            self.leftPointFile = open(self.leftPointsPath, 'a')
            self.rightPointFile = open(self.rightPointsPath, 'a')
            self.leftPointFile.write('\n')
            self.rightPointFile.write('\n')

    # text box on the right of slider bar
    def sliderValue(self):
        alphaValue = self.Slider.value() / 20
        self.alphaShow.setText('{0:2.2f}'.format(alphaValue))

    # check if both images are loaded. If so, enable rest of the functionality of the app
    def btnEnable(self):
        if self.leftLoaded and self.rightLoaded:
            self.Slider.setDisabled(False)
            self.btnBlend.setDisabled(False)
            if self.correspondence:
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
        return [], 0, False
    # read from file
    with open(filePath) as file:
        lines = file.readlines()
    points = []
    for eachLine in lines:
        data = eachLine.split()
        if data:
            data = [float(data[0]), float(data[1])]
            points.append(data)
    return points, len(points), True


# takes triangle's vertices and image size returns lines to be plotted
def getLines(vertices):
    lines = [QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1]),
             QtWidgets.QGraphicsLineItem(vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1]),
             QtWidgets.QGraphicsLineItem(vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1])]
    for each in lines:
        each.setPen(QtGui.QPen(QtCore.Qt.red, 1))
    return lines


# save the file immediately after write
def fileSave(file):
    file.flush()
    os.fsync(file.fileno())


if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()
