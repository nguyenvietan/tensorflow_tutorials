import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


a = tf.placeholder(tf.float32)
b = tf.constant(10, tf.float32)
list_vals = [1, 2, 3, 4, 5]
c = a + b

with tf.Session() as sess:
	for i in list_vals:
		print sess.run(c, feed_dict={a: i})
	

"""
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b

with tf.Session() as sess:
	print sess.run(c, {a:[1, 2, 3]})

"""








"""
w = tf.get_variable('scalar', initializer=tf.constant(1))
with tf.Session() as sess:
	sess.run(w.initializer)
	print w.eval()
	#sess.run(tf.global_variables_initializer())
	#print sess.run(w.assign_add(10))
	#print sess.run(w.assign_sub(2))
"""	

"""
x = tf.get_variable('number', initializer=tf.constant(1))

assign_op = x.assign(x*2)

with tf.Session() as sess:
	sess.run(tf.variables_initializer([x]))
	for i in range(10):
		sess.run(assign_op)
		print x.eval()
"""

"""
s = tf.get_variable('scalar', initializer=tf.constant(111))
m = tf.get_variable('matrix', initializer=tf.constant([[1, 2], [3, 4]]))
w = tf.get_variable('big_matrix', shape=(20,10), initializer=tf.zeros_initializer())

assign_op = s.assign(100)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print w.eval()
	#sess.run(tf.variables_initializer([s, m]))
	#print sess.run(w.initializer)	
	sess.run(assign_op)	
	print s.eval()	
"""


"""
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

r = tf.range(1,10,1)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
	print sess.run([x, r])
writer.close()
"""
