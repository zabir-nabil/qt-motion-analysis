# import all the necessary modules
# @author : zabir-nabil
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

from PoseEstimationModel import *
from PoseEstimationProcessing import *

pe = PoseEstimationModel('model.h5') # https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
pemodel = pe.create_model() # create the model

import cv2
import matplotlib
import pylab as plt
import numpy as np

processor = PoseEstimationProcessing() # load the processor

from PlotPoints import *
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('dance_motion.mp4')

ax1 = plt.subplot(1,1,1)

plt.title('Tappware Body Motion Application')

im1 = ax1.imshow(np.zeros((224,224,3)))

ret = True
cnt = 0
plt.ion()
while(ret):
    ret, frame = vid.read()

    if cnt % 25 == 0: # to counter the lagging effect
        a = cv2.resize(frame, (224,224))
        shared_pts = processor.shared_points(pemodel, a)
        print('Body parts location: ')
        print(shared_pts)
        plot = plot_circles(a, shared_pts)
        
        im1.set_data(plot)
        plt.pause(0.001)

        

        
    cnt += 1

plt.ioff()
plt.show()