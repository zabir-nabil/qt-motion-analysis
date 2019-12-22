# GUI related
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets                     
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                             QLabel, QVBoxLayout, QPlainTextEdit)            

from PyQt5.QtGui import QIcon, QPixmap

from gui import Ui_Form      

# import all the necessary modules
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
import scipy
import math
# import the pose estimation model and processing module
from PoseEstimationModel import *
from PoseEstimationProcessing import *

# extra backend dependency
import matplotlib
import pylab as plt
import numpy as np

# global scope
cur_id = 0
shared_locs = None

# load the pre-trained weights
pe = PoseEstimationModel('model.h5') 
pemodel = pe.create_model() # create the model

# skeleton model plotting helper function

def plot_skeleton(canvas, subject_wise_loc):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    cmap = matplotlib.cm.get_cmap('hsv')

    #canvas = cv2.imread(test_image)
    stickwidth = 4

    for i in range(len(subject_wise_loc)):

        for n in range(len(subject_wise_loc[i])):
            
            cur_canvas = canvas.copy()
            Y = subject_wise_loc[i][n][1]
            X = subject_wise_loc[i][n][0]

            mX = np.mean(X)
            mY = np.mean(Y)

            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.circle(canvas, (int(mY),int(mX)), 10, colors[i], thickness=-1)

            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

class video (QtWidgets.QDialog, Ui_Form):
    def __init__(self):
        super().__init__()                  

        self.setupUi(self)                                     

        self.control_bt.clicked.connect(self.start_webcam)
        self.capture.clicked.connect(self.capture_image)
        self.capture.clicked.connect(self.startUIWindow)       

        self.image_label.setScaledContents(True)

        self.cap = None                                        

        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0

    @QtCore.pyqtSlot()
    def start_webcam(self):
        if self.cap is None:
            # getting webcam ID
            # assuming the second parameter is Webcam ID
            # print(sys.argv)
            WEBCAM_ID = 0
            if len(sys.argv) > 1:
                WEBCAM_ID = int(sys.argv[1]) 
            self.cap = cv2.VideoCapture(WEBCAM_ID)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.timer.start()

    @QtCore.pyqtSlot()
    def update_frame(self):
        ret, image = self.cap.read()
        simage     = cv2.flip(image, 1)
        self.displayImage(image, True)

    @QtCore.pyqtSlot()
    def capture_image(self):
        flag, frame = self.cap.read()
        path = r"logs"                       
        if flag:
            QtWidgets.QApplication.beep()
            global cur_id
            cur_id += 1
            name = "image" +  str(cur_id) +  ".jpg"
            cv2.imwrite(os.path.join(path, name), frame)
            self._image_counter += 1
            # processing starts
            processor = PoseEstimationProcessing() # load the processor
            shared_pts = processor.shared_points(pemodel, frame) # shared points across multiple subjects
            subject_wise_loc = processor.subject_points(shared_pts)
            subject_wise_loc = np.array(subject_wise_loc)
            print(subject_wise_loc)
            out_frame = plot_skeleton(frame, subject_wise_loc)
            out_name = "out" +  str(cur_id) +  ".jpg"
            global shared_locs
            shared_locs = subject_wise_loc # global sharing the points
            cv2.imwrite(os.path.join(path, out_name), out_frame)

    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))

    def startUIWindow(self):
        self.Window = UIWindow()                               
        self.setWindowTitle("Motion Analytics 2")

        self.Window.ToolsBTN.clicked.connect(self.goWindow1)

        self.hide()
        self.Window.show()

    def goWindow1(self):
        self.setWindowTitle("Log")
        self.show()
        self.Window.hide()



class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)

        self.resize(800, 800)
        self.label = QLabel("<< Log result of Captured frame >>", alignment=QtCore.Qt.AlignCenter)
        # show current processed frame
        global cur_id

        path = r"logs" 
        out_name = "out" +  str(cur_id) +  ".jpg"

        self.pixmap = QPixmap(os.path.join(path, out_name))
        self.label.setPixmap(self.pixmap)

        self.logbox = QPlainTextEdit(self)
        self.logbox.setReadOnly(True)
        ### log generation ###

        body_map = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19']
        log_str = '<< Log result of Captured frame >>\n'


        log_str += 'Frame ID: ' + str(cur_id) + '\n'

        global shared_locs
        sub_body_coords = shared_locs

        log_str += 'NUMBER OF SUBJECTS: ' + str(sub_body_coords.shape[1]) + '\n'

        log_str += 'Log data: '

        for i in range(sub_body_coords.shape[1]):
            for j in range(sub_body_coords.shape[0]):
                x_c = sub_body_coords[j,i][0][0]
                print(x_c)
                if x_c == -1.:
                    continue
                log_str += body_map[j] + '  found for subject ' + str(i+1) + ' at ' + '(' + str(sub_body_coords[j,i][0]) +' , ' + str(sub_body_coords[j,i][1]) +  ') ; '
            log_str += '\n'

        ### ###
        self.logbox.insertPlainText(log_str)
        self.logbox.move(10,10)
        self.logbox.resize(200,200)
        self.ToolsBTN = QPushButton('Go back to main App')

        self.v_box = QVBoxLayout()
        self.v_box.addWidget(self.label)
        self.v_box.addWidget(self.ToolsBTN)
        self.v_box.addWidget(self.logbox)
        self.setLayout(self.v_box)


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    print(sys.argv)
    window = video()
    window.setWindowTitle('Motion Analytics 2')
    window.show()
    sys.exit(app.exec_())