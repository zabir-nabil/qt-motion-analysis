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
import numpy as np
import util
import cv2

class PoseEstimationModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        
    def relu(self, x): 
        return Activation('relu')(x)

    def conv(self, x, nf, ks, name):
        x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
        return x1

    def pooling(self, x, ks, st, name):
        x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
        return x

    def vgg_block(self, x):

        # Block 1
        x = self.conv(x, 64, 3, "conv1_1")
        x = self.relu(x)
        x = self.conv(x, 64, 3, "conv1_2")
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool1_1")

        # Block 2
        x = self.conv(x, 128, 3, "conv2_1")
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv2_2")
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool2_1")

        # Block 3
        x = self.conv(x, 256, 3, "conv3_1")
        x = self.relu(x)    
        x = self.conv(x, 256, 3, "conv3_2")
        x = self.relu(x)    
        x = self.conv(x, 256, 3, "conv3_3")
        x = self.relu(x)    
        x = self.conv(x, 256, 3, "conv3_4")
        x = self.relu(x)    
        x = self.pooling(x, 2, 2, "pool3_1")

        # Block 4
        x = self.conv(x, 512, 3, "conv4_1")
        x = self.relu(x)    
        x = self.conv(x, 512, 3, "conv4_2")
        x = self.relu(x)    

        # Additional non vgg layers
        x = self.conv(x, 256, 3, "conv4_3_CPM")
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv4_4_CPM")
        x = self.relu(x)

        return x

    def stage1_block(self, x, num_p, branch):

        # Block 1        
        x = self.conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
        x = self.relu(x)
        x = self.conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
        x = self.relu(x)
        x = self.conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)

        return x

    def stageT_block(self, x, num_p, stage, branch):

        # Block 1        
        x = self.conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
        x = self.relu(x)
        x = self.conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))

        return x
    
    def create_model(self):
        input_shape = (None,None,3)

        img_input = Input(shape=input_shape)

        stages = 6
        np_branch1 = 38
        np_branch2 = 19

        img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

        # VGG
        stage0_out = self.vgg_block(img_normalized)

        # stage 1
        stage1_branch1_out = self.stage1_block(stage0_out, np_branch1, 1)
        stage1_branch2_out = self.stage1_block(stage0_out, np_branch2, 2)
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stageT_branch1_out = self.stageT_block(x, np_branch1, sn, 1)
            stageT_branch2_out = self.stageT_block(x, np_branch2, sn, 2)
            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
        model.load_weights(self.weights_path)
        
        # save the model
        self.model = model
        
        return model