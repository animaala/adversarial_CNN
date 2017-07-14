########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module handles network training**
#
# Implemented in Python 3.5, TF v1.1, CuDNN 5.1
#
# Ryan Halliburton 2017
#
# This project is available at the following 2 repositories:
# 1. https://github.com/animaala/adversarial_ML
# 2. https://gitlab.ncl.ac.uk/securitylab/adversarial_ML.git
#
########################################################################

import tensorflow as tf
import numpy as np
import trafficsign_input as input

########################################################################

# Various constants for describing the data set

# location of the data set: a TFRecord file
DATA_PATH = input.DATA_PATH

# number of classes is 2 (go and stop)
NUM_CLASSES = input.NUM_CLASSES

# Width and height of each image. (pixels)
WIDTH = input.WIDTH
HEIGHT = input.HEIGHT

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = input.NUM_CHANNELS

# batch size for training/validating network
BATCH_SIZE = input.BATCH_SIZE


def train():
    with tf.Graph().as_default():
        # Force input pipeline to CPU:0 to avoid ops sometimes ending up on GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            # Get images and labels for the traffic_sign data set.
            # at this point label is one hot vector. If label = 1 then [1,0]... if label = 2 then [0,1]
            image, label = input.get_image(DATA_PATH + "train-00000-of-00001")
            # chance to distort the image
            image = input.distort_colour(image)
            # and similarly for the validation data
            vimage, vlabel = input.get_image(DATA_PATH + "validation-00000-of-00001")
            vimage = input.distort_colour(vimage)

