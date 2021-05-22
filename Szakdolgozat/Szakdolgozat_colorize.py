import PyQt5

from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog,QColorDialog,QLabel
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush, QPixmap
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2
import opt_first_alg as cz
import opt_second_alg as cd
import chrom_blend as ch_blending
import os
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        title = "Semi-automatic Colorizaton"
        top = 400
        left = 400
        width = 400
        height = 400
        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)
        self.scratch = QImage(self.size(), QImage.Format_RGB32)
        self.image = self.scratch
        self.image2 = self.scratch
        self.image.fill(Qt.white)
        self.color_image = self.scratch
        self.drawing = False
        self.brushSize = 3
        self.brushColor = Qt.white
        self.lastPoint = QPoint()
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        brushSize = mainMenu.addMenu("Brush Size")
        brushColor = mainMenu.addMenu("Brush Color")
        openAction = QAction(QIcon("icons/open.png"), "New", self)
        openAction.setShortcut("Ctrl+O")
        fileMenu.addAction(openAction)
        openAction.triggered.connect(self.open)

        loadAction = QAction(QIcon("icons/open.png"), "Load", self)
        loadAction.setShortcut("Ctrl+L")
        fileMenu.addAction(loadAction)
        loadAction.triggered.connect(self.load)

        cancelAction = QAction(QIcon("icons/cancel.png"), "Cancel", self)
        cancelAction.setShortcut("Ctrl+Z")
        fileMenu.addAction(cancelAction)
        cancelAction.triggered.connect(self.cancelevent)

        saveAction = QAction(QIcon("icons/save.png"), "Save", self)
        saveAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        clearAction = QAction(QIcon("icons/clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        startFirstAction = QAction(QIcon("icons/start.png"), "Start First Colorization using Optimization", self)
        startFirstAction.setShortcut("Ctrl+F")
        fileMenu.addAction(startFirstAction)
        startFirstAction.triggered.connect(self.startFirst)

        startSecondAction = QAction(QIcon("icons/start.png"),"Start Second Colorization using Optimization", self)
        startSecondAction.setShortcut("Ctrl+R")
        fileMenu.addAction(startSecondAction)
        startSecondAction.triggered.connect(self.startSecond)

        startChroBlend = QAction(QIcon("icons/start.png"), "Start Chrominance Blending", self)
        startSecondAction.setShortcut("Ctrl+B")
        fileMenu.addAction(startChroBlend)
        startChroBlend.triggered.connect(self.ChromBlend)

        onepxAction = QAction(QIcon("icons/pixel.png"), "1px", self)
        brushSize.addAction(onepxAction)
        onepxAction.triggered.connect(self.onePixel)

        twopxAction = QAction(QIcon("icons/pixel.png"), "2px", self)
        brushSize.addAction(twopxAction)
        twopxAction.triggered.connect(self.twoPixel)

        threepxAction = QAction(QIcon("icons/pixel.png"), "3px", self)
        brushSize.addAction(threepxAction)
        threepxAction.triggered.connect(self.threePixel)

        fourpxAction = QAction(QIcon("icons/pixel.png"), "4px", self)
        brushSize.addAction(fourpxAction)
        fourpxAction.triggered.connect(self.fourPixel)

        fivepxAction = QAction(QIcon("icons/pixel.png"), "5px", self)
        brushSize.addAction(fivepxAction)
        fivepxAction.triggered.connect(self.fivePixel)

        sevenpxAction = QAction(QIcon("icons/pixel.png"), "7px", self)
        brushSize.addAction(sevenpxAction)
        sevenpxAction.triggered.connect(self.sevenPixel)

        ninepxAction = QAction(QIcon("icons/pixel.png"), "9px", self)
        brushSize.addAction(ninepxAction)
        ninepxAction.triggered.connect(self.ninePixel)

        ColorAction = QAction(QIcon("icons/color.png"), "Color", self)
        ColorAction.setShortcut("Ctrl+R")
        brushColor.addAction(ColorAction)
        ColorAction.triggered.connect(self.colorChooser)


        self.label = QLabel()
        self.setCentralWidget(self.label)


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.scratch.save("current/helper1.bmp")
            self.image2.save("current/helper2.bmp")
            self.drawing = True
            self.lastPoint = event.pos() + PyQt5.QtCore.QPoint(0, -22)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.scratch)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos()+ PyQt5.QtCore.QPoint(0, -22))
            self.lastPoint = event.pos() + PyQt5.QtCore.QPoint(0, -22)
            self.update()
            self.label.setPixmap(self.scratch)



    def mouseReleaseEvent(self, event):

        self.scratch.save("current/image_marked.bmp")
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())


    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "All Files(*.*);;PNG(*.png);;JPEG(*.jpg *.jpeg) ")

        if filePath == "":
            return
        self.scratch.save(filePath)

    def load(self):
        print("load")
        filePath, _ = QFileDialog.getOpenFileName(self, "Open original Image", "",
                                                  "All Files(*.*);;PNG(*.png);;BMP(*.bmp);;JPEG(*.jpg *.jpeg) ")
        print(filePath)
        if filePath == "":
            return

        self.original = QPixmap(filePath)
        self.label.setPixmap(self.original)
        self.resize(self.original.size())
        self.original.save("current/image.bmp")

        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image with scribbles", "",
                                                  "All Files(*.*);;PNG(*.png);;BMP(*.bmp);;JPEG(*.jpg *.jpeg) ")
        self.scratch = QPixmap(filePath)
        self.image2 = QImage(self.scratch.size(), QImage.Format_RGB32)

        self.adjustSize()
        self.update()
        self.label.setPixmap(self.scratch)

        self.scratch.save("current/image_marked.bmp")
        self.update()
    def cancelevent(self):
        self.image2 = QPixmap('current/helper2.bmp')
        self.scratch = QPixmap('current/helper1.bmp')
        self.label.setPixmap(self.scratch)

    def open(self):
        filePath, _ =  QFileDialog.getOpenFileName(self, "Open Image", "",
                                                  "All Files(*.*);;PNG(*.png);;BMP(*.bmp);;JPEG(*.jpg *.jpeg) ")
        if filePath == "":
            return
        self.scratch= QPixmap(filePath)
        self.image2 = QImage(self.scratch.size(), QImage.Format_RGB32)
        self.original = QPixmap(filePath)
        self.label.setPixmap(self.scratch)
        self.resize(self.scratch.size())
        self.original.save("current/image.bmp")

        self.adjustSize()
        self.update()
    def clear(self):
        self.image.fill(Qt.black)
        self.scratch=self.original
        self.label.setPixmap(self.scratch)
        self.update()


    def ChromBlend(self):
        print("StartSecond")
        print("Prepare data")
        dir_name = "done"
        dir_name_2 = "current"
        out_suffix = ".bmp"
        out_image_name = os.path.join(dir_name, "image" + "_Ch_blend" + out_suffix)
        out_image_name_v = os.path.join(dir_name_2, "image" + "_Ch_blend" + out_suffix)
        ch_blending.run_colorization(out_image_name,out_image_name_v)
        print("done")




    def startSecond(self):
        print("StartSecond")
        print("Prepare data")
        dir_name = "done"
        out_suffix = "_res.bmp"
        out_image_name = os.path.join(dir_name, "image" + "second_alg" + out_suffix)
        cd.run_colorization(out_image_name)
        print("done")

    def startFirst(self):
        print("StartFirst")
        print("Prepare data")
        dir_name = "current"
        iteration = 1
        threshold = 0.01
        wd = 1
        input_suffix = ".bmp"
        marked_suffix = "_marked.bmp"
        mask_suffix = "_mask.bmp"
        marked_intensity_suffix = "_marked_intensity.bmp"
        gray_suffix = "_gray.bmp"
        out_suffix = "_res.bmp"
        input_image_name = os.path.join(dir_name, "image" + input_suffix)
        gray_image_name = os.path.join(dir_name, "image" + gray_suffix)
        marked_image_name = os.path.join(dir_name, "image" + marked_suffix)
        parameter_suffix = "_iterations_" + str(iteration) + "_threshold_" + str(threshold) + "_wd_" + str(wd)
        marked_intensity_image_name = os.path.join(dir_name, "image" + parameter_suffix + marked_intensity_suffix)
        mask_image_name = os.path.join(dir_name, "image" + parameter_suffix + mask_suffix)
        dir_name = "done"
        out_image_name = os.path.join(dir_name, "image" + "first_alg" + out_suffix)
        input_image_uint8 = cv2.imread(input_image_name, cv2.IMREAD_COLOR)
        marked_image_uint8 = cv2.imread(marked_image_name, cv2.IMREAD_COLOR)
        input_image = input_image_uint8.astype(np.float) / 255
        marked_image = marked_image_uint8.astype(np.float) / 255
        res = cz.get_gray_mask(input_image, marked_image, threshold=threshold)
        gray_image = res["gray_image"]
        mask_image = res["mask_image"]
        marked_intensity_image = res["marked_intensity"]
        cv2.imwrite(gray_image_name, gray_image * 255)
        cv2.imwrite(mask_image_name, mask_image * 255)
        cv2.imwrite(marked_intensity_image_name, marked_intensity_image * 255)
        marked_image_yiq = res["marked_image_yiq"]
        mark = res["mark"]

        res = cz.colorize(marked_image_yiq, mark, iteration=iteration, wd=wd)

        self.color_image = res["color_image"]
        cv2.imshow('First Colorization using Optimization', self.color_image)
        cv2.imwrite(out_image_name, self.color_image * 255)

    def onePixel(self):
        self.brushSize = 1

    def twoPixel(self):
        self.brushSize = 2

    def threePixel(self):
        self.brushSize = 3

    def fourPixel(self):
        self.brushSize = 4

    def fivePixel(self):
        self.brushSize = 5

    def sevenPixel(self):
        self.brushSize = 7

    def ninePixel(self):
        self.brushSize = 9

    def colorChooser(self):
        self.brushColor = QColorDialog.getColor()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()