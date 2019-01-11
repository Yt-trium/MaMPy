#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
User interface for demonstration of max-tree and area open filter.
Implementation
C. Meyer
"""

import sys
import numpy as np
import qimage2ndarray

# Qt5 includes
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog
from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QSpinBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore

# MaMPy includes
# Utilities
from utils import image_read
# Max-Tree Berger
from algo2 import maxtree_union_find_level_compression, compute_attribute_area, direct_filter


class MaMPyGUIDemoMaxTree(QMainWindow):
    # default size of GUI
    xsize = 800
    ysize = 600

    # Images Attributes
    imageSrc = None
    imageSrcFlat = None
    maxtree_p = None
    maxtree_s = None
    maxtree_a = None

    # Initialisation of the class
    def __init__(self):
        super().__init__()
        self.initUI()

    # Initialisation of the user interface.
    def initUI(self):
        # Image Label Source
        self.imageLabelSrc = QLabel()
        self.imageLabelSrc.setAlignment(QtCore.Qt.AlignCenter)
        # Image Label Result
        self.imageLabelRes = QLabel()
        self.imageLabelRes.setAlignment(QtCore.Qt.AlignCenter)

        # Area Threshold Slider
        self.areaThresholdSlider = QSlider(QtCore.Qt.Horizontal)
        self.areaThresholdSlider.setRange(0, 1)
        self.areaThresholdSlider.valueChanged.connect(self.areaThresholdSliderChanged)
        # Area Threshold Spinbox
        self.areaThresholdSpinbox = QSpinBox()
        self.areaThresholdSpinbox.setRange(0, 1)
        self.areaThresholdSpinbox.valueChanged.connect(self.areaThresholdSpinboxChanged)

        # Layout
        self.imageLabelLayout = QHBoxLayout()
        self.thresholdLayout = QHBoxLayout()

        self.imageLabelLayout.addWidget(self.imageLabelSrc)
        self.imageLabelLayout.addWidget(self.imageLabelRes)
        self.imageLabelLayoutWidget = QWidget()
        self.imageLabelLayoutWidget.setLayout(self.imageLabelLayout)

        self.thresholdLayout.addWidget(self.areaThresholdSlider)
        self.thresholdLayout.addWidget(self.areaThresholdSpinbox)
        self.thresholdLayoutWidget = QWidget()
        self.thresholdLayoutWidget.setLayout(self.thresholdLayout)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.imageLabelLayoutWidget)
        self.mainLayout.addWidget(self.thresholdLayoutWidget)

        # Main Widget
        self.window = QWidget()
        self.window.setLayout(self.mainLayout)

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
        self.setWindowTitle('MaMPy Max-Tree demo')
        self.show()

    def resizeEvent(self, event):
        self.xsize = event.size().width()
        self.ysize = event.size().height()

        self.updateImages()

        QMainWindow.resizeEvent(self, event)

    def areaThresholdSliderChanged(self, event):
        self.areaThresholdSpinbox.setValue(self.areaThresholdSlider.value())
        self.updateImages()

    def areaThresholdSpinboxChanged(self, event):
        self.areaThresholdSlider.setValue(self.areaThresholdSpinbox.value())
        self.updateImages()

    def openImageDialog(self):
        filename = QFileDialog.getOpenFileName(self, "Open file", None, 'Image Files (*.png *.jpg *.bmp)')

        if filename[0]:
            # Read the image
            self.imageSrc = image_read(filename[0])

            # Compute the Max-Tree
            (self.maxtree_p, self.maxtree_s) = maxtree_union_find_level_compression(self.imageSrc, connection8=True)

            self.imageSrcFlat = self.imageSrc.flatten()
            self.maxtree_a = compute_attribute_area(self.maxtree_s, self.maxtree_p, self.imageSrcFlat)

            # Update slider and spinbox in the image resolution range
            self.areaThresholdSlider.setRange(0, (self.imageSrc.shape[0] * self.imageSrc.shape[1]))
            self.areaThresholdSpinbox.setRange(0, (self.imageSrc.shape[0] * self.imageSrc.shape[1]))

            # Update the image label
            self.updateImages()

    def updateImages(self):
        # Check if an image is loaded
        if self.imageSrc is None:
            return

        # Image Source
        pixmapSrc = QPixmap(qimage2ndarray.array2qimage(self.imageSrc))
        pixmapSrc = pixmapSrc.scaled((self.xsize / 2) - 50, self.ysize - 50, QtCore.Qt.KeepAspectRatio)

        self.imageLabelSrc.setPixmap(pixmapSrc)

        # Image Result
        self.imageRes = direct_filter(self.maxtree_s, self.maxtree_p, self.imageSrcFlat, self.maxtree_a, self.areaThresholdSpinbox.value())
        self.imageRes = self.imageRes.reshape(self.imageSrc.shape)

        pixmapRes = QPixmap(qimage2ndarray.array2qimage(self.imageRes))
        pixmapRes = pixmapRes.scaled((self.xsize / 2) - 50, self.ysize - 50, QtCore.Qt.KeepAspectRatio)

        self.imageLabelRes.setPixmap(pixmapRes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MaMPyGUIDemoMaxTree()
    sys.exit(app.exec_())
