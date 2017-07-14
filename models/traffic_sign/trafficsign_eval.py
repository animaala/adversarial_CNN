########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module handles evaluation, accuracy and summary logging**
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

########################################################################


def visualize_kernel(W):
    with tf.variable_scope('kernel_visualisation'):
        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(W)
        x_max = tf.reduce_max(W)
        W_0_to_1 = (W - x_min) / (x_max - x_min)

        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])

        # this will display random 3 filters from the 128 in conv1
        tf.summary.image('conv1', kernel_transposed, 3)