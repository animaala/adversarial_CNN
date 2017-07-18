########################################################################
#
# A Convolutional Neural model binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module handles the model training**
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
import trafficsign_model as model
import trafficsign_image_processing as input
import os
import random

########################################################################

# Various constants for describing the data set

# location of the data set: a TFRecord file
DATA_PATH = model.DATA_PATH

# Width and height of each image. (pixels)
HEIGHT = model.HEIGHT
WIDTH = model.WIDTH

# number of classes is 2 (go and stop)
NUM_CLASSES = model.NUM_CLASSES

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = model.NUM_CHANNELS

########################################################################

X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS], name="images")
# similarly, we have a placeholder for true outputs (obtained from labels)
Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")
# variable learning rate
lr = tf.placeholder(tf.float32, name="learning_rate")
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.5 at training time
pkeep = tf.placeholder(tf.float32, name="dropout_prob")


with tf.device('/cpu:0'):
    imageBatch, labelBatch = input.distorted_image_batch(DATA_PATH+"train-00000-of-00001")


# Build a Graph that computes the logits predictions from the inference model.
logits = model.inference(X, pkeep)
Y = tf.nn.softmax(logits)
# compute cross entropy loss on the logits
loss = model.loss(logits, Y_)
# get accuracy of logits with the ground truth
accuracy = model.accuracy(logits, Y_)
# Build a Graph that trains the model with one batch of examples and updates the model parameters.
train_step = model.optimize(loss, lr)


# image_buffer, label = input.parse_single_image(DATA_PATH + "train/stop/" + file_name)
# image = input.decode_jpeg(image_buffer)

file_name = random.choice(os.listdir(DATA_PATH + "train/stop/"))
image_data = tf.gfile.FastGFile(DATA_PATH + "train/stop/" + file_name, 'rb').read()

image = input.decode_jpeg(image_data)

image = tf.expand_dims(image, 0)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


batch_xs = sess.run(image)
predictions = sess.run(Y, {X: batch_xs, pkeep:1.0})
print(predictions)


# finalise
coord.request_stop()
coord.join(threads)