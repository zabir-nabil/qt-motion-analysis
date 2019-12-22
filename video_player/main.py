from time import sleep
import cv2
import sys
from threading import Timer,Thread,Event
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QVBoxLayout
from PyQt5 import QtGui, QtCore ,QtWidgets

from MainForm import Ui_MainWindow
import numpy as np
from openpyxl import load_workbook,Workbook
import pandas as pd
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from weakref import ref
import pynput
import os

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


from PlotPoints import *

from PoseEstimationModel import *
from PoseEstimationProcessing import *

from globals import *

class perpetualTimer():

   def __init__(self,t,hFunction):
      self.t=t
      self.hFunction = hFunction
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()



class workerThread2(QThread):
    #updatedLine = QtCore.pyqtSignal(int)
    def __init__(self,mw):
        self.mw=mw
        QThread.__init__(self)
    def __del__(self):
        self.wait()

    def run(self):        
        
        #QApplication.processEvents()
        while self.mw.isRun:
            
            if self.mw.isthreadActive:
                #cnt=self.mw.stframe + self.mw.ui.horizontalSlider.value()-1
                cnt=self.mw.frameID
                if cnt+1-self.mw.stframe<self.mw.ui.horizontalSlider.maximum() and not self.mw.sliderbusy and not self.mw.resizegoing:            
            
                    #self.mw.ui.horizontalSlider.setValue(cnt+1-self.mw.stframe)

                    pdraw=self.mw.drawmin
                    while (cnt+1)/self.mw.fps>self.mw.data.values[self.mw.drawmin,1]:
                        self.mw.drawmin+=1
                    if not self.mw.drawmin==pdraw:        
                        #print(self.mw.drawmin)
                        wr = ref(self.mw.ax1.lines[5])
                        self.mw.ax1.lines.remove(wr())
                        self.mw.ax1.plot([self.mw.data.values[self.mw.drawmin,1],self.mw.data.values[self.mw.drawmin,1]],[0,3.4],'k',linewidth=2.0)
                        self.mw.ui.bottomImage.canvas.draw()
                        QApplication.processEvents()
                        #sleep(0.01)
                    else:
                        sleep(0.01)
                else:
                    sleep(0.01)
            else:
                sleep(0.01)

            

    

class workerThread(QThread):
        
    updatedM = QtCore.pyqtSignal(int)

    

    def __init__(self,mw):
        self.mw=mw
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        
        itr=0
        QApplication.processEvents()
        while self.mw.isRun:
            itr+=1
            
            if self.mw.isthreadActive and  self.mw.isbusy==False and self.mw.frameID != self.mw.cap.get(cv2.CAP_PROP_POS_FRAMES):               
                #print(itr)

                if  np.abs(self.mw.frameID-self.mw.cap.get(cv2.CAP_PROP_POS_FRAMES))>1:
                     self.mw.cap.set(cv2.CAP_PROP_POS_FRAMES,self.mw.frameID)

                if self.mw.timer is None:
                    self.mw.frameID+=1

                self.mw.isbusy=True              
                sf=self.mw.scaleFactor 
                if proc_frames.get(self.mw.frameID) is not None:

                    ret = True
                    image = proc_frames[self.mw.frameID]
                else:
                    ret, image = self.mw.cap.read() 
                self.mw.limg=image
                if sf<=0:
                    self.mw.isbusy=False
                    continue
                if ret==False:
                      self.mw.isthreadActiv=False
                      self.mw.isbusy=False
                      continue              
                                
                nchannel=image.shape[2]

                limg2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
                 
                timg=cv2.resize(limg2,(int(sf*limg2.shape[1]),int(sf*limg2.shape[0]))) 
                limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel*timg.shape[1], QtGui.QImage.Format_RGB888)  
                if self.mw.resizegoing==False:
                    self.mw.ui.LeftImage.setPixmap(QtGui.QPixmap(limage))
                    if not self.mw.sliderbusy and not self.mw.sliderbusy2:
                        self.updatedM.emit(self.mw.frameID)                    

                    QApplication.processEvents()
                self.mw.isbusy=False 
            else:
                if self.mw.isthreadActive and self.mw.timer is None:
                    self.mw.frameID+=1
                sleep(1.0/50)


class MainForm(QMainWindow):

    resized = QtCore.pyqtSignal()
    #keypressed = QtCore.pyqtSignal()

    def __init__(self):
        super(MainForm,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        
        layout = QVBoxLayout()
        
        self.figure = Figure()
        self.ui.bottomImage.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.ui.bottomImage.canvas)        
        self.ui.bottomImage.setLayout(layout)

        self.ax1 = self.figure.add_subplot(1,1,1 )#, projection='3d')
        
        self.ax1.get_yaxis().set_visible(False)
        self.ax1.get_xaxis().set_visible(True)
        
        self.figure.subplots_adjust(left=0.001, right=0.999, top=1.0, bottom=0.1)



        self.scaleFactor=1.0
        self.frameID=0        
        self.strt=0
        self.isRun=True
        self.resizegoing=False
        self.sliderbusy=False
        self.sliderbusy2=False
        self.linebusy=False
        self.template=np.zeros((1,1,1))
        self.ui.openButton.clicked.connect(self.openButtonPressed)
        self.ui.startButton.clicked.connect(self.startButtonPressed)   
        self.ui.fileButton.clicked.connect(self.fileButtonPressed)      
        self.ui.horizontalSlider.sliderPressed.connect(self.horizontalSliderPressed)
        self.ui.horizontalSlider.sliderReleased.connect(self.horizontalSliderReleased)
        self.ui.horizontalSlider.valueChanged.connect(self.slider_value_changed)
        self.fps = 20

        # processed frames


        self.ui.pauseButton.clicked.connect(self.pauseButtonPressed)
        #self.ui.MainWindow.
        
        self.resized.connect(self._on_resized)
        #self.keypressed.connect(self._key_pressed)
        self.startx=0
        self.starty=0
        self.isvideo=False
        self.isdata=False

        
        

        self.isbusy=0
        self.frameHeight=1
        self.frameWidth=1
        self.limg=np.zeros((1,1,1))
        self.tracker=None
        self.cap=None
        self.timer=None

       
        self.ui.statusbar.showMessage("Select a video for analysis")
        self.isthreadActive=False

        self.wthread = workerThread(self)        
        self.wthread.updatedM.connect(self.horizontalSliderSet)        
        self.wthread.start()  

        self.wthread2 = workerThread2(self)        
        #self.wthread2.updatedLine.connect(self.lineSliderSet)       
        self.wthread2.start() 

        klistener=pynput.keyboard.Listener(on_press=self.on_press) #,on_release=self.on_release)
        klistener.start()

        #self.keythread=MyWidget(self)


    #def on_release(self,key):
    #    if self.isRun:
    #        if key==pynput.keyboard.Key.left:
    #            print('left')
    #            #self.horizontalSliderIncrease(1)
    #        elif key==pynput.keyboard.Key.right:
    #            print('right')
    #            #self.horizontalSliderIncrease(-1)

    def on_press(self,key):        
         if self.isRun:            
            if pynput.keyboard.Key.space==key: 
                if self.ui.pauseButton.isEnabled():
                    self.pauseButtonPressed()
                else:
                    while (self.sliderbusy==True or self.resizegoing==True):
                        sleep(0.1)
                    self.startButtonPressed()

    def slider_value_changed(self):        
        if not self.isthreadActive:
            #print('slidervalue change')
            self.horizontalSliderIncrease(0)
           
    def horizontalSliderIncrease(self,val):
        if self.sliderbusy or self.resizegoing:        
            return
        self.sliderbusy=True
        #print(self.ui.horizontalSlider.value())
        #self.ui.horizontalSlider.setValue(self.ui.horizontalSlider.value()+val)
        #print(self.frameID)
        self.frameID=self.stframe + self.ui.horizontalSlider.value()-1  
        #print(self.frameID)     
        #self.drawmin=1
        if self.ui.startButton.isEnabled():
                      
            if proc_frames.get(self.frameID) is not None:
                frame = proc_frames[self.frameID]
            else: 
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frameID)        
                ret, frame = self.cap.read()
            self.limg=frame            
            #self.on_zoomfit_clicked()
            nchannel=frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                  
            timg=cv2.resize(limg2,(int(self.scaleFactor*limg2.shape[1]),int(self.scaleFactor*limg2.shape[0]))) 
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel*timg.shape[1], QtGui.QImage.Format_RGB888)  
            self.ui.LeftImage.setPixmap(QtGui.QPixmap(limage))
            self.lineSliderSet(self.frameID)

        self.sliderbusy=False   

    def updateFrame(self):
            self.frameID+=1
            
    def lineSliderSet(self,cnt):
        if cnt+1-self.stframe>self.ui.horizontalSlider.maximum():
            return

        self.linebusy=True
        pdraw=self.drawmin

        self.drawmin-=20
        if self.drawmin<1:
            self.drawmin=1
        while (cnt+1)/self.fps>self.data.values[self.drawmin,1]:
            self.drawmin+=1
        if not self.drawmin==pdraw or pdraw==1:     
           
            wr = ref(self.ax1.lines[-1])
            self.ax1.lines.remove(wr())
            self.ui.bottomImage.canvas.draw()
            self.ax1.plot([self.data.values[self.drawmin,1],self.data.values[self.drawmin,1]],[0,20],'k',linewidth=2.0)
            self.ui.bottomImage.canvas.draw()

        tsec=cnt/self.fps
        tmin=int(tsec/60)
        ttsec=int(tsec-60*tmin)
        ksec=tsec-60*tmin-ttsec
                

        self.ui.statusbar.showMessage("To analyze a frame, press Process Frame and wait")

        self.linebusy=False

    def horizontalSliderSet(self,cnt):
        if cnt+1-self.stframe>self.ui.horizontalSlider.maximum() or self.sliderbusy or self.resizegoing:
            return
        self.sliderbusy2=True        
        self.ui.horizontalSlider.setValue(cnt+1-self.stframe)  
        tsec=cnt/self.fps
        tmin=int(tsec/60)
        ttsec=int(tsec-60*tmin)
        ksec=tsec-60*tmin-ttsec
                

        self.ui.statusbar.showMessage("To analyze a frame, press Process Frame and wait")      
        self.sliderbusy2=False
        
    
    def horizontalSliderPressed(self):
        self.sliderbusy=True

    def horizontalSliderReleased(self):
        self.frameID=self.stframe + self.ui.horizontalSlider.value()-1
        
        self.drawmin=1
        if self.ui.startButton.isEnabled():
            print(type(proc_frames.get(self.frameID)))
            print(proc_frames.get(self.frameID))
            if proc_frames.get(self.frameID) is not None:
                frame = proc_frames[self.frameID]
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frameID)           
                ret, frame = self.cap.read()


            # real time update
            print(self.frameID)
            
            self.limg=frame            
            self.on_zoomfit_clicked()
            nchannel=frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                  
            timg=cv2.resize(limg2,(int(self.scaleFactor*limg2.shape[1]),int(self.scaleFactor*limg2.shape[0]))) 
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel*timg.shape[1], QtGui.QImage.Format_RGB888)  
            self.ui.LeftImage.setPixmap(QtGui.QPixmap(limage))
            self.lineSliderSet(self.frameID)

        self.sliderbusy=False

    def resizeEvent(self, event):
        self.resized.emit()
        return super(MainForm, self).resizeEvent(event)

    def _on_resized(self):
        self.on_zoomfit_clicked()
   

    def on_zoomfit_clicked(self):        

        self.resizegoing=True
        a=self.ui.LeftImage.size()
        if a.width()/self.frameWidth < a.height()/self.frameHeight:
            self.scaleFactor=a.width()/self.frameWidth
            self.startx=0
            self.starty= (a.height() - self.scaleFactor * self.frameHeight)/2
        else:
            self.scaleFactor=1.0*a.height()/self.frameHeight
            self.starty=0
            self.startx= (a.width() - self.scaleFactor * self.frameWidth)/2.0
        sleep(0.2)
        self.resizegoing=False

    def openButtonPressed(self):
        if self.isthreadActive:
            return

        self.ui.statusbar.showMessage("Wait while our machine learning model processes the video") 

        self.pe = PoseEstimationModel('model.h5') # https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
        self.pemodel = self.pe.create_model() # create the model

        self.processor = PoseEstimationProcessing()

        



        import time
        import random
        import json


        # random processing


        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frameID)           
        ret, frame = self.cap.read()
        a = cv2.resize(frame, (224,224))
        shared_pts = self.processor.shared_points(self.pemodel, a)

        print(type(shared_pts))
        
        a = plot_circles(a, shared_pts)

        a2 = cv2.resize(a, (self.frameWidth, self.frameHeight))
        proc_frames[self.frameID] = a2[:]
        print('Updated %d'%(self.frameID))
        print(proc_frames[self.frameID])

        # log generation


        with open('logs.db', 'a') as outfile:
            outfile.write("frame_no: %d frame_pts_data: "%(self.frameID))
            outfile.write(str(shared_pts))

        self.limg= a2       
        self.on_zoomfit_clicked()
        nchannel=a2.shape[2]
        limg2 = cv2.cvtColor(a2, cv2.COLOR_RGB2BGR)                  
        timg=cv2.resize(limg2,(int(self.scaleFactor*limg2.shape[1]),int(self.scaleFactor*limg2.shape[0]))) 
        limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel*timg.shape[1], QtGui.QImage.Format_RGB888)  
        self.ui.LeftImage.setPixmap(QtGui.QPixmap(limage))
        self.lineSliderSet(self.frameID)





    def fileButtonPressed(self):

        fileName = QFileDialog.getOpenFileName(None,caption="Select Video File",directory=QtCore.QDir.currentPath())
        try:
            print('1')

            if len(fileName[0])>0:
                self.cap = cv2.VideoCapture(fileName[0])
                self.isvideo=True
            else:
                print('No video updated')
                return

            # cnt = 0
            # test_cap = cv2.VideoCapture(fileName[0])
            # rett, framet = test_cap.read()
            # out_img = framet
            # n_im = 0
            # while(rett):
            #     rett, framet = test_cap.read()
            #     if cnt%20 == 0:
            #         out_img = np.concatenate((out_img,framet), axis=1)
            #         n_im += 1
                
            #     if n_im == 15:
            #         break
                
            #     cnt += 1

            # frame_bar = out_img # cv2.resize(out_img, (100,20))

            print('2')

            length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(length)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print('FPS:')
            print(self.fps)


            print('3')

            self.data = pd.read_excel('output.xlsx')  
            self.stframe=int(20.125*self.fps)    
            self.endframe=int(89*self.fps)  

            self.ui.horizontalSlider.setMaximum(self.endframe-self.stframe)

            self.cap.set(1,self.stframe)
            ret, frame=self.cap.read()



            print('4')
            self.drawmin=1
            self.frameID=self.stframe
            self.limg=frame
            print('5')
            self.frameHeight=frame.shape[0]
            self.frameWidth=frame.shape[1] 
            self.on_zoomfit_clicked()

            print('6')

            nchannel=frame.shape[2]
            limg2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                    
            timg=cv2.resize(limg2,(int(self.scaleFactor*limg2.shape[1]),int(self.scaleFactor*limg2.shape[0]))) 
            limage = QtGui.QImage(timg.data, timg.shape[1], timg.shape[0], nchannel*timg.shape[1], QtGui.QImage.Format_RGB888)  
            self.ui.LeftImage.setPixmap(QtGui.QPixmap(limage))

            self.ui.statusbar.showMessage("Ready to Start, or redefine data or video") 
            self.ui.startButton.setEnabled(True)
            self.ui.pauseButton.setEnabled(False)
            self.ui.horizontalSlider.setEnabled(True)


                    



            print(self.stframe)
            print(self.endframe)

            print(self.data)       
            self.isdata=True      
            self.draw()  
            self.ui.statusbar.showMessage("Select Video File")  
            self.ui.openButton.setEnabled(True)
        except Exception as e: print(e)

        # fileName = QFileDialog.getOpenFileName(None,caption="Select Data File in Excel",directory=QtCore.QDir.currentPath())
        # if len(fileName[0])>0:
        #  try:
        #    book = load_workbook(fileName[0])
        #    ws = book.active
        #    self.data = pd.DataFrame(ws.values)     
        #    print(self.data)       
        #    self.isdata=True      
        #    self.draw()  
        #    self.ui.statusbar.showMessage("Select Video File")  
        #    self.ui.openButton.setEnabled(True)
        #  except:
        #    pass

        # else:
        #    return

    def draw(self):              
        
        self.ax1.clear()        
        #self.ax.axis('off')
        
        data=self.data.values[1:,1:5]
        data[:,1]= (data[:,1]-data[:,1].min())/(data[:,1].max()-data[:,1].min())
        data[:,2]= (data[:,2]-data[:,2].min())/(data[:,2].max()-data[:,2].min())
        data[:,3]= (data[:,3]-data[:,3].min())/(data[:,3].max()-data[:,3].min())
        #self.ax.set_xlim([-0.5,0.5])
        self.ax1.set_ylim([0,20])
        #self.ax1.imshow(frame_bar) 
        #xi = cv2.imread('icon.png')
        #xi = cv2.resize(xi, (100,5))
        
        #self.ax.set_zlim([-0.5,0.5])
        self.ax1.plot([1,2,3],[4,5,6],'y',label=self.data.values[0,2])
        self.ax1.plot([1,2,3],[4,5,6],'r',label=self.data.values[0,3])
        #self.ax1.plot(data[:,0],data[:,3]+0.1,'g',label=self.data.values[0,4])
        self.ax1.plot([1,2,3],[4,5,6],'g',label=self.data.values[0,4])

        self.ax1.plot([0,1] ,[1,1],'k')
        self.ax1.plot([0,1] ,[1,1],'k')
        

        self.ax1.set_xlim([20,86])       

        self.ax1.plot([self.data.values[1,1],self.data.values[1,1]],[0,20],'k',linewidth=2.0)   
        
  
        start, end = self.ax1.get_xlim()
        self.ax1.xaxis.set_ticks([])        
        #self.ax1.legend(loc='best',  framealpha=0.1,ncol=3)
  
        
        self.ui.bottomImage.canvas.draw()

    def startButtonPressed(self):         
        
        if self.isthreadActive:
            return             
                     

        self.ui.startButton.setEnabled(False)
        print(len(proc_frames))

        if self.ui.timeComboBox.currentIndex()==0:
            self.timer = perpetualTimer(1.0/self.fps,self.updateFrame)
            self.timer.start()
        else:
            self.timer=None
        
        self.ui.pauseButton.setEnabled(True)
        self.isthreadActive=True

    def pauseButtonPressed(self):        

         if not self.isthreadActive:
            return 
         self.ui.startButton.setEnabled(True)         
         self.ui.pauseButton.setEnabled(False)
         if not self.timer is None:
            self.timer.cancel()
         self.isthreadActive=False


os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
app=QApplication(sys.argv)
app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
widget=MainForm()
widget.show()
sys.exit(app.exec_())
klistener.join()