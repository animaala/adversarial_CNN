import tensorflow as tf


X = tf.constant(["./adversarial_image/stop_normal.jpg"])





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./traffic_graph/test", sess.graph)
    print(sess.run(z))
