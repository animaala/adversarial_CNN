import tensorflow as tf
import trafficsign_input as input


image, label = input.get_image("../../dataset/traffic_sign/")

image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=64,
    capacity=200,
    min_after_dequeue=20)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(label_batch)))
