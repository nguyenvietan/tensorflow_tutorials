import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import utils

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


#
learning_rate = 0.01
batch_size = 128
n_epochs = 30

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)
tf.logging.set_verbosity(old_v)

X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# build model
logits = tf.matmul(X, w) + b  # [128x10]

# define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name='loss') # [128]
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch   # [] scalar

# Define training op
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# calculate accuracy
preds = tf.nn.softmax(logits) #[128x10]. logits -> exp(logits)/reduce_sum(logits, axis)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1)) # (preds == Y)
accuracy = tf.reduce_sum(tf.cast(correct_preds, dtype=tf.float32))


# a = tf.constant([[1, 2, 3], 
# 				[4, 5, 6], 
# 				[7, 8, 9]], dtype=tf.float32, name='a')
# b = tf.constant([[1, 2, 3], 
# 				[4, 5, 4], 
# 				[7, 8, 9]], dtype=tf.float32, name='a')

# b = tf.nn.softmax(a) 
# c0 = tf.argmax(a, 0) # [2 2 2]
# c1 = tf.argmax(a, 1) # [2 2 2]



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	n_batch_train = int(mnist.train.num_examples/batch_size)
	print "n_batches_train = ", n_batch_train
	# train
	for i in range(n_epochs):
		total_loss = 0
		for j in range(n_batch_train):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			_, l = sess.run([optimizer, loss], {X: X_batch, Y: Y_batch})
			total_loss += l
		print "epoch {0}: loss {1}".format(i, total_loss/n_batch_train)
			
	total_correct = 0

	# accurary
	n_batch_test = int(mnist.test.num_examples/batch_size)

	for i in range(n_batch_test):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, {X: X_batch, Y: Y_batch})
		total_correct += accuracy_batch

	# print "n_batch_test ", n_batch_test
	# print "total accuracy: ", total_correct
	# print mnist.test.num_examples
	print "accuracy rate = ", total_correct/mnist.test.num_examples


















