########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **Create adversarial example from a randomly selected image**
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
import trafficsign_image_processing as input
import trafficsign_model as model
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # filter out INFO, keep WARNING+

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
    cls_target_pl = tf.placeholder(dtype=tf.int32)
    tf.summary.image("image", X, 3)

########################################################################


def find_adversary_noise(decoded_img, cls_target=1, noise_limit=4.0, required_score=0.95, max_iterations=100):
    """Find the noise that must be added to the given
    image so that it is classified as the target-class.
    :param image: Input image which will be altered into adversarial example. Must have dtype=int32
    shape [height, width, channels]
    :param cls_target: The target class we wish the image to become. Our classes are go:1 and stop:2.
    However the softmax_cross_entropy functions require 0 and 1.. so minus 1 from each class number.
    i.e. class go=0 and class stop=1
    :param noise_limit: Limit for pixel-values in the noise
    :param required_score: Stop when target-class confidence reaches this
    :param max_iterations: Max number of optimization iterations to perform
    :return:
    """

    label = tf.stack(tf.one_hot(cls_target-1, NUM_CLASSES))
    label = tf.reshape(label, [1, 2])
    tgt_label = label.eval()

    # input image is [h,w,channels]. Expand dimensions so it's [1,h,w,ch] so it can be fed to inference
    image = tf.expand_dims(decoded_img, 0).eval()

    # Calculate the predicted class-scores (aka. probabilities)
    pred = Y.eval(feed_dict={X: image, pkeep: 1.0})

    # Convert to one-dimensional array. i.e. from 'softmax:0' -> [[  5.19299786e-17   1.00000000e+00]]
    pred = np.squeeze(pred)                # to                  [  5.19299786e-17   1.00000000e+00]

    # Predicted class-number, argmax returns 0 or 1 so +1 for 1 or 2, which is our class numbers
    cls_source = np.argmax(pred) + 1
    # Score for the predicted class (aka. probability or confidence).
    score_source_org = pred.max()
    print("Original image is predicted as Class {} with a confidence of {}".format(cls_source, score_source_org))

    # Names for the source and target classes.
    name_source = "stop"
    name_target = "go"

    # Initialize the noise to zero.
    noise = 0

    # Perform a number of optimization iterations to find
    # the noise that causes mis-classification of the input image.
    for i in range(max_iterations):
        print("Iteration:", i)

        # The noisy image is just the sum of the input image and noise.
        noisy_image = image+noise

        # Ensure the pixel-values of the noisy image are between
        # 0 and 255 like a real image. If we allowed pixel-values
        # outside this range then maybe the mis-classification would
        # be due to this 'illegal' input breaking the model.
        noisy_image = tf.clip_by_value(tf.to_int32(noisy_image), 0, 255).eval()

        # Calculate the predicted class-scores as well as the gradient.
        pred = Y.eval(feed_dict={X: noisy_image, pkeep: 1.0})


        loss = sess.run(adv_loss, feed_dict={X: noisy_image, pkeep: 1.0, Y_: tgt_label})
        print("The current loss is: {}".format(loss))

        grad = sess.run(gradient, feed_dict={X: noisy_image, pkeep: 1.0, Y_: tgt_label})


        # Convert the predicted class-scores to a one-dim array.
        pred = np.squeeze(pred)

        print("Current Prediction for noisy image: {}".format(pred))

        # The scores (probabilities) for the source and target classes.
        # -1 to get back in the range 0 or 1, (our classes are 1 and 2)
        score_source = pred[cls_source-1]
        score_target = pred[cls_target-1]

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now tells us how much we need to change the
        # noisy input image in order to move the predicted class
        # closer to the desired target-class.

        # Calculate the max of the absolute gradient values.
        # This is used to calculate the step-size.
        grad_absmax = np.abs(grad).max()
        print("Gradient abs Max: {}".format(grad_absmax))

        # If the gradient is very small then use a lower limit,
        # because we will use it as a divisor.
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # Calculate the step-size for updating the image-noise.
        # This ensures that at least one pixel colour is changed by 7.
        # Recall that pixel colours can have 255 different values.
        # This step-size was found to give fast convergence.
        step_size = 8/grad_absmax

        # Print the score etc. for the source-class.
        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        # Print the score etc. for the target-class.
        msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        # Print statistics for the gradient.
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))

        # Newline.
        print()

        # If the score for the target-class is not high enough.
        if score_target < required_score:
            # Update the image-noise by subtracting the gradient
            # scaled by the step-size.
            noise -= step_size*grad

            # Ensure the noise is within the desired range.
            # This avoids distorting the image too much.
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            # Abort the optimization because the score is high enough.
            break

    return image.squeeze(), noisy_image.squeeze(), noise, \
           name_source, name_target, \
           score_source, score_source_org, score_target



with tf.device('/cpu:0'):
    imageBatch, labelBatch = input.distorted_image_batch(DATA_PATH+"train-00000-of-00001")
    vimageBatch, vlabelBatch = input.distorted_image_batch(DATA_PATH+"validation-00000-of-00001")


# get references to various model component tensors
logits = model.inference(X, pkeep)
Y = tf.nn.softmax(logits)
# compute cross entropy loss on the logits
model_loss = model.loss(logits, Y_)
# get accuracy of logits with the ground truth
accuracy = model.accuracy(logits, Y_)
# Build a Graph that trains the model with one batch of examples and updates the model parameters.
train_step = model.optimize(model_loss, lr)


adv_loss = model.loss(logits, Y_, mean=False)
gradient = tf.gradients(adv_loss, X)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# start training
nSteps = 400
for i in range(nSteps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # train_step is the backpropagation step. Running this op allows the network to learn the distribution of the data
    train_step.run(feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})

    if (i+1)%50 == 0:  # work out training accuracy and log summary info for tensorboard
        train_acc = sess.run(accuracy, feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})
        print("step {}, training accuracy {}".format(i+1, train_acc))


image, label = input.get_stop_image("stop (286).jpg")
image, adv_img, noise, name_src, name_target, score_src, score_source_org, score_target = find_adversary_noise(image)

image = tf.expand_dims(image, 0)
adv_img = tf.expand_dims(adv_img, 0)

img_pred = Y.eval(feed_dict={X:image.eval(), pkeep:1.0})
adv_pred = Y.eval(feed_dict={X:adv_img.eval(), pkeep:1.0})

print("Original Image is classed as: Class {}".format(img_pred))
print("Adversarial_Image is classed as: Class {}".format(adv_pred))

# finalise
coord.request_stop()
coord.join(threads)
