import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# read data
DATA_FILE = 'data/birth_life_2010.txt'
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# create placeholders
X = tf.placeholder(tf.float32)
y_d = tf.placeholder(tf.float32)

# create variables
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Model
y_predicted = X*w + b
loss = tf.square(y_predicted - y_d)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100):
		total_loss = 0
		for x, y in data:
			_, l = sess.run([optimizer, loss], feed_dict={X: x, y_d: y})
			total_loss += l
		print('Epoch {0}: {1}').format(i, total_loss/n_samples)

	w_out, b_out = [w.eval(), b.eval()]
	
print w_out, b_out

plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0]*w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()

