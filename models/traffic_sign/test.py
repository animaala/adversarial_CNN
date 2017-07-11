import tensorflow as tf


t = tf.Variable(tf.range(1, 6913, 1))
init = tf.global_variables_initializer()
t = tf.reshape(t, [6, 6, 3, 64])

with tf.Session() as sess:
    sess.run(init)
    W1_a = t  # [6, 6, 3, 64]
    W1_b = tf.split(W1_a, 64, 3)  # 64 x [6, 6, 3, 1]
    print(tf.shape(W1_b))
    W1_row0 = tf.concat(W1_b[0:8], 0)  # 8 x [5, 5, 3, 1]
    W1_row1 = tf.concat(W1_b[8:16], 0)  # 8 x [5, 5, 3, 1]
    W1_row2 = tf.concat(W1_b[16:24], 0)  # 8 x [5, 5, 3, 1]
    W1_row3 = tf.concat(W1_b[24:32], 0)  # 8 x [5, 5, 3, 1]
    W1_row4 = tf.concat(W1_b[32:40], 0)  # 8 x [5, 5, 3, 1]
    W1_row5 = tf.concat(W1_b[40:48], 0)  # 8 x [5, 5, 3, 1]
    W1_row6 = tf.concat(W1_b[48:56], 0)  # [30, 5, 3, 1]
    W1_row7 = tf.concat(W1_b[56:64], 0)  # [30, 5, 3, 1]

    W1_d = tf.concat([W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5, W1_row6, W1_row7], 1)  # [30, 30, 3, 1]
    print(tf.shape(W1_d))
    W1_e = tf.reshape(W1_d, [64, 6, 6, 3])
    tf.summary.image(W1_e)



