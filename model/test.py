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



# Build a Graph that computes the logits predictions from the inference model.
logits = model.inference(X, pkeep)
Y = tf.nn.softmax(logits)
# compute cross entropy loss on the logits
loss = model.loss(logits, Y_)
# get accuracy of logits with the ground truth
accuracy = model.accuracy(logits, Y_)
# Build a Graph that trains the model with one batch of examples and updates the model parameters.
train_step = model.optimize(loss, lr)


# # we have a second loss function to find the gradient towards the adversarial example
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")
adv_train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


with tf.device('/cpu:0'):
    imageBatch, labelBatch = input.distorted_image_batch(DATA_PATH+"train-00000-of-00001")


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# start training
nSteps = 50
for i in range(nSteps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # train_step is the backpropagation step. Running this op allows the network to learn the distribution of the data
    train_step.run(feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})

    if (i+1)%50 == 0:  # work out training accuracy and log summary info for tensorboard
        train_acc = sess.run(accuracy, feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})
        print("step {}, training accuracy {}".format(i+1, train_acc))



image_data, l = input.parse_single_image()
image = input.decode_jpeg(image_data)
image = tf.expand_dims(image, 0)
batch_xs = sess.run(image)
preds = Y.eval(feed_dict={X: batch_xs, pkeep:1.0})
print(preds)


label = tf.stack(tf.one_hot(1-1, NUM_CLASSES))
label = tf.reshape(label, [1, 2])
label = label.eval()
print(label)

# loss = .eval(feed_dict={X: batch_xs, pkeep:1.0, Y_: label})
print(loss)

# finalise
coord.request_stop()
coord.join(threads)
