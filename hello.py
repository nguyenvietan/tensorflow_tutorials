import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

r = tf.range(1,10,1)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
	print sess.run([x, r])

writer.close()

