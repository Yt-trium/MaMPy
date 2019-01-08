#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

# Qt5 includes
from PyQt5.QtWidgets import QMainWindow, QLabel, QAction, QFileDialog, QApplication, QGridLayout, QPushButton, QWidget, QSlider
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore

class MaMPyGUI(QMainWindow):
    # size of GUI
    xsize = 800
    ysize = 600

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Image Label Source
        self.imageLabelSrc = QLabel()
        self.imageLabelSrc.setMinimumSize(self.xsize/3, self.ysize/3)
        self.imageLabelSrc.setAlignment(QtCore.Qt.AlignCenter)
        # Image Label Result
        self.imageLabelRes = QLabel()
        self.imageLabelRes.setMinimumSize(self.xsize/3, self.ysize/3)
        self.imageLabelRes.setAlignment(QtCore.Qt.AlignCenter)

        # Area Threshold
        self.areaThreshholdSlider = QSlider(QtCore.Qt.Horizontal)
        self.areaThreshholdSlider.setRange(0, 1)

        # Compute Button
        self.computeButton = QPushButton("Compute")
        self.computeButton.clicked.connect(self.computeOpenArea)

        # Main Layout
        self.centerLayout = QGridLayout()
        self.centerLayout.addWidget(self.imageLabelSrc, 0, 0)
        self.centerLayout.addWidget(self.imageLabelRes, 0, 1)
        self.centerLayout.addWidget(self.areaThreshholdSlider, 1, 0)
        self.centerLayout.addWidget(self.computeButton, 1, 1)

        # Main Widget
        self.window = QWidget()
        self.window.setLayout(self.centerLayout)

        # Set Central Widget
        self.setCentralWidget(self.window)

        # Actions
        # Open File
        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.triggered.connect(self.openImageDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        # Center Window
        xbase = (app.desktop().screenGeometry().width()  - self.xsize) / 2
        ybase = (app.desktop().screenGeometry().height() - self.ysize) / 2

        # Window Property
        self.setGeometry(xbase, ybase, self.xsize, self.ysize)
        self.setWindowTitle('MaMPy')
        self.show()

    def openImageDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            pixmap = QPixmap(fname[0])
            pixmap = pixmap.scaled(self.xsize/2, self.ysize, QtCore.Qt.KeepAspectRatio)
            self.imageLabelSrc.setPixmap(pixmap)
            self.imageLabelRes.setPixmap(pixmap)

    def computeOpenArea(self):
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MaMPyGUI()
    ex = MaMPyGUI()
    sys.exit(app.exec_())
