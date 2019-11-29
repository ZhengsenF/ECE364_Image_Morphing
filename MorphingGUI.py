# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MorphingGUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(660, 700)
        self.btnLoadLeft = QtWidgets.QPushButton(Dialog)
        self.btnLoadLeft.setGeometry(QtCore.QRect(10, 10, 171, 32))
        self.btnLoadLeft.setObjectName("btnLoadLeft")
        self.btnLoadRight = QtWidgets.QPushButton(Dialog)
        self.btnLoadRight.setGeometry(QtCore.QRect(330, 10, 161, 32))
        self.btnLoadRight.setObjectName("btnLoadRight")
        self.imageLeftViewer = QtWidgets.QGraphicsView(Dialog)
        self.imageLeftViewer.setGeometry(QtCore.QRect(10, 50, 301, 221))
        self.imageLeftViewer.setObjectName("imageLeftViewer")
        self.imageRight = QtWidgets.QGraphicsView(Dialog)
        self.imageRight.setGeometry(QtCore.QRect(330, 50, 311, 221))
        self.imageRight.setObjectName("imageRight")
        self.Slider = QtWidgets.QSlider(Dialog)
        self.Slider.setGeometry(QtCore.QRect(80, 350, 481, 22))
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setObjectName("Slider")
        self.checkBox = QtWidgets.QCheckBox(Dialog)
        self.checkBox.setGeometry(QtCore.QRect(270, 290, 111, 20))
        self.checkBox.setObjectName("checkBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(90, 290, 91, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(460, 290, 91, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 350, 60, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(80, 370, 60, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(550, 370, 60, 16))
        self.label_5.setObjectName("label_5")
        self.alphaShow = QtWidgets.QLineEdit(Dialog)
        self.alphaShow.setGeometry(QtCore.QRect(580, 350, 61, 21))
        self.alphaShow.setObjectName("alphaShow")
        self.imageBlend = QtWidgets.QGraphicsView(Dialog)
        self.imageBlend.setGeometry(QtCore.QRect(180, 390, 301, 221))
        self.imageBlend.setObjectName("imageBlend")
        self.btnBlend = QtWidgets.QPushButton(Dialog)
        self.btnBlend.setGeometry(QtCore.QRect(260, 650, 171, 32))
        self.btnBlend.setObjectName("btnBlend")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(290, 630, 101, 16))
        self.label_6.setObjectName("label_6")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btnLoadLeft.setText(_translate("Dialog", "Load Starting Image ..."))
        self.btnLoadRight.setText(_translate("Dialog", "Load Ending Image ..."))
        self.checkBox.setText(_translate("Dialog", "Show Triangle"))
        self.label.setText(_translate("Dialog", "Starting Image"))
        self.label_2.setText(_translate("Dialog", "Ending Image"))
        self.label_3.setText(_translate("Dialog", "Alpha"))
        self.label_4.setText(_translate("Dialog", "0.0"))
        self.label_5.setText(_translate("Dialog", "1.0"))
        self.btnBlend.setText(_translate("Dialog", "Blend"))
        self.label_6.setText(_translate("Dialog", "Blending Image"))
