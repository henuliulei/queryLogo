import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from compute import Detection
import cv2


class picture(QWidget):
    path1 = ""
    path2 = ""
    def __init__(self):
        super(picture, self).__init__()

        self.resize(1200, 600)
        self.setWindowTitle("单样本深度学习检测效果演示")

        self.label = QLabel(self)
        self.label.setFixedSize(250, 300)
        self.label.move(300, 130)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label1 = QLabel(self)
        self.label1.setFixedSize(250, 300)
        self.label1.move(600, 130)
        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        self.label2 = QLabel(self)
        self.label2.setFixedSize(250, 300)
        self.label2.move(900, 130)
        self.label2.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )


        self.label3 = QLabel(self)
        self.label3.setFixedSize(150, 100)
        self.label3.move(350, 50)
        self.label3.setText(" 子图像")
        self.label3.setStyleSheet(
                                  "QLabel{color:rgb(300,300,300,120);font-size:30px;font-family:宋体;}"
                                  )

        self.label4 = QLabel(self)
        self.label4.setFixedSize(150, 100)
        self.label4.move(650, 50)
        self.label4.setText("目标图像")
        self.label4.setStyleSheet(
            "QLabel{color:rgb(300,300,300,120);font-size:30px;font-family:宋体;}"
        )
        self.label5 = QLabel(self)
        self.label5.setFixedSize(150, 100)
        self.label5.move(950, 50)
        self.label5.setText("定位结果")
        self.label5.setStyleSheet(
            "QLabel{color:rgb(300,300,300,120);font-size:30px;font-family:宋体;}"
        )

        btn = QPushButton(self)
        btn.setText("打开子图像")
        btn.move(50, 80)
        btn.setFixedSize(200, 100)
        btn.clicked.connect(self.openimage)

        btn1 = QPushButton(self)
        btn1.setText("打开目标图")
        btn1.move(50, 230)
        btn1.setFixedSize(200, 100)
        btn1.clicked.connect(self.openimage1)

        btn2 = QPushButton(self)
        btn2.setText("定位")
        btn2.move(50, 380)
        btn2.setFixedSize(200, 100)

        btn2.clicked.connect(self.detect)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "./yanshi/", "*.jpg;;*.png;;All Files(*)")
        picture.path1 =imgName
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
    def openimage1(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "./yanshi/", "*.jpg;;*.png;;All Files(*)")
        picture.path2 = imgName
        print( picture.path2 )
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label1.setPixmap(jpg)
    def detect(self):
        print(picture.path1,picture.path2)
        if picture.path1!="" and picture.path2 !="":
            xmin, ymin, xmax, ymax= Detection.getPredict(picture.path1, picture.path2)
            print(xmin, ymin, xmax, ymax)
            self.jpg = cv2.imread(picture.path2)
            if xmin != -1:
                self.cvRGBImg = cv2.rectangle(self.jpg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 10)
            else:
                self.cvRGBImg = cv2.putText(self.jpg, "no logo detected", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (100, 50, 200), 6)
            self.img_rgb = cv2.cvtColor(self.cvRGBImg, cv2.COLOR_BGR2BGRA)
            self.QtImg = QtGui.QImage(self.img_rgb.data, self.img_rgb.shape[1], self.img_rgb.shape[0],QtGui.QImage.Format_RGB32)
            self.jpg = QtGui.QPixmap(self.QtImg).scaled(self.label2.width(), self.label2.height())
            self.label2.setPixmap(self.jpg)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
