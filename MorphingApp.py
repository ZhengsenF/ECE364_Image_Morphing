#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 25
# #######################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from MorphingGUI import *


class MorphingApp(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MorphingApp, self).__init__(parent)
        self.setupUi(self)
        self.alphaShow.setDisabled(True)



if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()
