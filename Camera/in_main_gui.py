# coding:utf-8

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qtawesome

from time import sleep, ctime
import numpy as np
import sys
import cv2

from net import Classifier
from Camera.in_GUI_init_layout import Initor_for_event

class MainUi(Initor_for_event):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setWindowTitle("汪星人识别系统")
        self.resize(1200, 910)
        self.video_size = (500, 360)
        self.det = Classifier()
        self.setFixedSize(self.width(), self.height())
        self.pred = None
        self.detect_bool = False
        self.expression = ''
        self.init_layout()
        self.init_clik()

    def init_clik(self):
        
        self.left_close.clicked.connect(self.close_all)

    def close_all(self):
        self.close()

    def init_layout(self):

        self.init_left()
        self.init_right()
        self.init_bottom_box()
        self.init_btn_event()
        self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.main_layout.setSpacing(0)
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)


    def load_local_image_file(self):

        image, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "*.jpg;;*.png;;All Files(*)")
        if image != "":  # 为用户取消
            print(image)
            im = cv2.imdecode(np.fromfile(image, dtype=np.uint8), 1)  # 读成彩图
            
            frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(frame.data,  width, height, bytesPerLine,
                             QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
            self.raw_video.setPixmap(QPixmap.fromImage(q_image))

            pred = self.det.test(image)

            frame = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(frame.data,  width, height, bytesPerLine,
                             QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
            self.show_video.setPixmap(QPixmap.fromImage(q_image))

