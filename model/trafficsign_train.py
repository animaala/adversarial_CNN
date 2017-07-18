########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module has functions to perform a single training step on the network
#   and a function to create an adversarial example**
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
    cls_target = tf.placeholder(dtype=tf.int32)
    tf.summary.image("image", X, 3)

########################################################################

def train(X, Y_, pkeep, lr):
    """Performs one training step with a batch of images and labels. Returns ops
    train_step and accuracy.
    :param X: 4-D tensor. Batch of images of shape [BATCH_SIZE, 72, 72, 3]
    :param Y_: 2-D tensor. Batch of labels of shape [BATCH_SIZE, 2]
    :param pkeep: Float. Probability of dropout. i.e. 0.5
    :param lr: Float. Learning rate. i.e. 0.0008
    :returns:
        train_step: op to run the training step
        accuracy: op to calculate the accuracy
    """
    # Build a Graph that computes the logits predictions from the inference model.
    logits = model.inference(X, pkeep)
    # compute cross entropy loss on the logits
    loss = model.loss(logits, Y_)
    # get accuracy of logits with the ground truth
    accuracy = model.accuracy(logits, Y_)
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_step = model.optimize(loss, lr)
    return train_step, accuracy


########################################################################


# sess = tf.InteractiveSession()
#
# # Use spare CPU cycles to retrieve batches of training and test images and an img for adversarial training
# with tf.device('/cpu:0'):
#     imageBatch, labelBatch = model.distorted_image_batch(DATA_PATH+"train-00000-of-00001")
#     vimageBatch, vlabelBatch = model.distorted_image_batch(DATA_PATH+"validation-00000-of-00001")
#     image, label = model.get_adversarial_image()
#
#
# # our model for classifying images
# logits = model.model(X, pkeep)
# Y = tf.nn.softmax(logits)
#
# # we have a second loss function to find the gradient towards the adversarial example
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[cls_target])
#
# # Get the gradient for the loss-function with regard to input image.
# gradient = tf.gradients(loss, image)
#
#
# def find_adversary_noise(image, cls_target=1, noise_limit=3.0, required_score=0.99, max_iterations=100):
#     """Find the noise that must be added to the given
#     image so that it is classified as the target-class.
#     :param image: Input image which will be altered into adversarial example
#     :param cls_target: The target class we wish the image to become
#     :param noise_limit: Limit for pixel-values in the noise
#     :param required_score: Stop when target-class confidence reaches this
#     :param max_iterations: Max number of optimization iterations to perform
#     :return:
#     """
#
#     # Calculate the predicted class-scores (aka. probabilities)
#     Ypred = sess.run(Y, feed_dict={X: image, pkeep: 1.0})
#
#     # Convert to one-dimensional array.
#     pred = np.squeeze(Ypred)
#
#     # Predicted class-number.
#     cls_source = np.argmax(pred)
#
#     # Score for the predicted class (aka. probability or confidence).
#     score_source_org = pred.max()
#
#     # Names for the source and target classes.
#     name_source = model.name_lookup.cls_to_name(cls_source,
#                                                 only_first_name=True)
#     name_target = model.name_lookup.cls_to_name(cls_target,
#                                                 only_first_name=True)
#
#     # Initialize the noise to zero.
#     noise = 0
#
#     # Perform a number of optimization iterations to find
#     # the noise that causes mis-classification of the input image.
#     for i in range(max_iterations):
#         print("Iteration:", i)
#
#         # The noisy image is just the sum of the input image and noise.
#         noisy_image = image+noise
#
#         # Ensure the pixel-values of the noisy image are between
#         # 0 and 255 like a real image. If we allowed pixel-values
#         # outside this range then maybe the mis-classification would
#         # be due to this 'illegal' input breaking the Inception model.
#         noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)
#
#         # Create a feed-dict. This feeds the noisy image to the
#         # tensor in the graph that holds the resized image, because
#         # this is the final stage for inputting raw image data.
#         # This also feeds the target class-number that we desire.
#         feed_dict = {model.tensor_name_resized_image:noisy_image,
#                      pl_cls_target:cls_target}
#
#         # Calculate the predicted class-scores as well as the gradient.
#         pred, grad = session.run([y_pred, gradient],
#                                  feed_dict=feed_dict)
#
#         # Convert the predicted class-scores to a one-dim array.
#         pred = np.squeeze(pred)
#
#         # The scores (probabilities) for the source and target classes.
#         score_source = pred[cls_source]
#         score_target = pred[cls_target]
#
#         # Squeeze the dimensionality for the gradient-array.
#         grad = np.array(grad).squeeze()
#
#         # The gradient now tells us how much we need to change the
#         # noisy input image in order to move the predicted class
#         # closer to the desired target-class.
#
#         # Calculate the max of the absolute gradient values.
#         # This is used to calculate the step-size.
#         grad_absmax = np.abs(grad).max()
#
#         # If the gradient is very small then use a lower limit,
#         # because we will use it as a divisor.
#         if grad_absmax < 1e-10:
#             grad_absmax = 1e-10
#
#         # Calculate the step-size for updating the image-noise.
#         # This ensures that at least one pixel colour is changed by 7.
#         # Recall that pixel colours can have 255 different values.
#         # This step-size was found to give fast convergence.
#         step_size = 7/grad_absmax
#
#         # Print the score etc. for the source-class.
#         msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
#         print(msg.format(score_source, cls_source, name_source))
#
#         # Print the score etc. for the target-class.
#         msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
#         print(msg.format(score_target, cls_target, name_target))
#
#         # Print statistics for the gradient.
#         msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
#         print(msg.format(grad.min(), grad.max(), step_size))
#
#         # Newline.
#         print()
#
#         # If the score for the target-class is not high enough.
#         if score_target < required_score:
#             # Update the image-noise by subtracting the gradient
#             # scaled by the step-size.
#             noise -= step_size*grad
#
#             # Ensure the noise is within the desired range.
#             # This avoids distorting the image too much.
#             noise = np.clip(a=noise,
#                             a_min=-noise_limit,
#                             a_max=noise_limit)
#         else:
#             # Abort the optimization because the score is high enough.
#             break
#
#     return image.squeeze(), noisy_image.squeeze(), noise, \
#            name_source, name_target, \
#            score_source, score_source_org, score_target


with tf.device('/cpu:0'):
    imageBatch, labelBatch = input.distorted_image_batch(DATA_PATH+"train-00000-of-00001")
    vimageBatch, vlabelBatch = input.distorted_image_batch(DATA_PATH+"validation-00000-of-00001")


#train_step, accuracy = train(X, Y_, pkeep, lr)

# Build a Graph that computes the logits predictions from the inference model.
logits = model.inference(X, pkeep)
print(logits)
Y = tf.nn.softmax(logits)
# compute cross entropy loss on the logits
loss = model.loss(logits, Y_)
# get accuracy of logits with the ground truth
accuracy = model.accuracy(logits, Y_)
# Build a Graph that trains the model with one batch of examples and updates the model parameters.
train_step = model.optimize(loss, lr)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# start training
nSteps = 100
for i in range(nSteps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # train_step is the backpropagation step. Running this op allows the network to learn the distribution of the data
    sess.run([train_step], feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})

    if (i+1)%50 == 0:  # work out training accuracy and log summary info for tensorboard
        train_acc = sess.run(accuracy, feed_dict={X:batch_xs, Y_:batch_ys, lr:0.0008, pkeep:0.5})
        print("step {}, training accuracy {}".format(i+1, train_acc))


file_name = random.choice(os.listdir(DATA_PATH+"train/stop/"))
image_data = tf.gfile.FastGFile(DATA_PATH+"train/stop/"+file_name, 'rb').read()

image = input.decode_jpeg(image_data)
image = tf.expand_dims(image, 0)


batch_xs = sess.run(image)
predictions = sess.run(Y, {X: batch_xs, pkeep:1.0})
print(predictions)



# finalise
coord.request_stop()
coord.join(threads)
