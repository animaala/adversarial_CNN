########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
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


# Placeholders for data we will populate later
with tf.name_scope("inputs"):
    # input X: 72*72*3 pixel images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS], name="images")
    # similarly, we have a placeholder for true outputs (obtained from labels)
    Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32, name="learning_rate")
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.5 at training time
    pkeep = tf.placeholder(tf.float32, name="dropout_prob")
    # placeholder for the adversarial target class, which will be 1, i.e. "go"
    pl_cls_target = tf.placeholder(dtype=tf.int32)
    tf.summary.image("image", X, 3)

with tf.name_scope("normal_model"):
    # get batches of images for training network
    with tf.name_scope("get_batch"):
        with tf.device('/cpu:0'):
            imageBatch, labelBatch = model.distorted_inputs(DATA_PATH + "train-00000-of-00001")
            vimageBatch, vlabelBatch = model.distorted_inputs(DATA_PATH + "validation-00000-of-00001")



    # Build a Graph that computes the logits predictions from the inference model.
    logits = model.inference(X, pkeep)

    # compute cross entropy loss on the logits
    loss = model.loss(logits, Y_)

    # get accuracy of logits with the ground truth
    accuracy = model.accuracy(logits, Y_)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_step = model.train(loss, lr)


# adversarial part of the graph
with tf.name_scope("adversarial"):

    with tf.device('/cpu:0'):
        image, label = model.adversarial_input()

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])





# interactive session allows interleaving of building and running steps
sess = tf.InteractiveSession()
# init
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

writer = tf.summary.FileWriter("./traffic_graph/main", sess.graph)
summary = tf.summary.merge_all()


# start training
nSteps = 1000
for i in range(nSteps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # train_step is the backpropagation step. Running this op allows the network to learn the distribution of the data
    sess.run([train_step], feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.0008, pkeep: 0.5})

    if (i + 1) % 50 == 0:  # work out training accuracy and log summary info for tensorboard
        train_acc, s = sess.run([accuracy, summary], feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.0008, pkeep: 0.5})
        writer.add_summary(s, i)
        print("step %d, training accuracy %g" % (i + 1, train_acc))

    if (i + 1) % 100 == 0:  # then perform validation

        # get a validation batch and calculate the accuracy of the predictions
        # note accuracy.eval() is equivalent to sess.run(accuracy)
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        test_acc = accuracy.eval(feed_dict={X: vbatch_xs, Y_: vbatch_ys, lr: 0.0008, pkeep: 1.0})

        print("step %d, test accuracy %g" % (i + 1, test_acc))