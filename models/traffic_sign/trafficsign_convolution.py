########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module handles the representation of the network**
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


def _activation_summary(x):
    """Helper to create activation summaries. Creates a summary that provides a histogram
    of activations. Creates a summary that measures the sparsity of activations.
    :param x: Tensor
    :return: Nothing
    """
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))


def inference(images, pkeep):
    """The model:
    2 conv layers, kernel shape [filter_height, filter_width, in_channels, out_channels]
    and 2 fully connected layers, one to bring all the activation maps together (outputs
    of all the filters) and one final softmax layer to predict the class.
    :param images: 4-D Tensor of images [batch, height, width, channels]
    :param pkeep: Dropout probability
    :return: logits
    """
    with tf.variable_scope("the_model"):
        # 2 convolutional layers with their channel counts, and a
        # fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
        J = 128   # 1st convolutional layer output channels
        K = 172   # 2nd convolutional layer output channels
        N = 1536  # fully connected layer

        # weights / kernels
        # 7x7 patch, 3 input channel, J output channels
        W1 = tf.Variable(tf.truncated_normal([7, 7, NUM_CHANNELS, J], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([5, 5, J, K], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([8 * 8 * K, N], stddev=0.1))
        W4 = tf.Variable(tf.truncated_normal([N, NUM_CLASSES], stddev=0.1))

        # biases
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [J]))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [NUM_CLASSES]))

        #visualize_kernel(W1)

        with tf.name_scope("first_layer"):
            # 72x72 images
            Y1r = tf.nn.relu(tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
            # 3x3 pooling area with stride of 3 reduces the image by 2/3 = 24x24 images after max_pool
            Y1p = tf.nn.max_pool(Y1r, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
            Y1 = tf.nn.dropout(Y1p, pkeep)
            _activation_summary(Y1)

        with tf.name_scope("second_layer"):
            Y2r = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
            # 3x3 pooling area with stride 3 reduces image by 2/3 = 8x8
            Y2p = tf.nn.max_pool(Y2r, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
            Y2 = tf.nn.dropout(Y2p, pkeep)
            _activation_summary(Y2)

        with tf.name_scope("fc_layer"):
            YY = tf.reshape(Y2, shape=[-1, 8 * 8 * K])
            Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)
            _activation_summary(Y3)

            YY4 = tf.nn.dropout(Y3, pkeep)
            Ylogits = tf.matmul(YY4, W4) + B4
            _activation_summary(Ylogits)
        # note we dont return softmax here.. only the unscaled logits.
        return Ylogits


def loss(logits, Y_):
    """Computes cross entropy loss on the unscaled logits from model.
    Add summary for cross entropy.
    :param logits: Logits from inference().
    :param Y_: one-hot label tensor
    :return: Loss tensor of type float
    """
    with tf.name_scope("x-ent"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
        cross_entropy = tf.reduce_mean(cross_entropy) * BATCH_SIZE
        tf.summary.scalar("x-ent", cross_entropy)
    return cross_entropy


def accuracy(logits, Y_):
    """Computes the accuracy of predictions.
    :param logits: Logits from inference().
    :param Y_: one-hot label tensor
    :return:
    """
    with tf.name_scope("accuracy"):
        Y = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    return accuracy