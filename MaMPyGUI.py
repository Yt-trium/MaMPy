#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

# Qt5 includes
from PyQt5.QtWidgets import QMainWindow, QLabel, QAction, QFileDialog, QApplication, QGridLayout, QPushButton, QWidget, QSlider
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor
from PyQt5 import QtCore

# Max Tree Berger includes
from algo2 import  maxtree_union_find_level_compression, compute_attribute_area, image_read, direct_filter

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
            file_path = fname[0]
            self.imageSrc = image_read(filename=file_path)

            self.flatten_image = self.imageSrc.flatten()
            (self.maxtree_parents, self.maxtree_s) = maxtree_union_find_level_compression(self.imageSrc, connection8=True)
            self.attr = compute_attribute_area(self.maxtree_s, self.maxtree_parents, self.flatten_image)

            out = direct_filter(self.maxtree_s, self.maxtree_parents, self.flatten_image, self.attr, 1000)
            out = np.reshape(out, self.imageSrc.shape)

            img = QImage(out, out.shape[1], out.shape[0], QImage.Format_Grayscale8)

            pixmap2 = QPixmap(img)

            pixmap = QPixmap(fname[0])
            pixmap = pixmap.scaled(self.xsize/2, self.ysize, QtCore.Qt.KeepAspectRatio)
            pixmap2 = pixmap2.scaled(self.xsize/2, self.ysize, QtCore.Qt.KeepAspectRatio)

            self.imageLabelSrc.setPixmap(pixmap)
            self.imageLabelRes.setPixmap(pixmap2)

    def computeOpenArea(self):
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MaMPyGUI()
    ex = MaMPyGUI()
    sys.exit(app.exec_())
