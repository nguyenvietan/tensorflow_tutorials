a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [12, 11, 10]], dtype=tf.float32, name='a')
b = tf.nn.softmax(a) 
c0 = tf.argmax(a, 0)
c1 = tf.argmax(a, 1)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run(c0)
	print sess.run(c1)



