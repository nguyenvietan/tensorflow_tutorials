import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import utils

#
learning_rate = 0.01
batch_size = 128
n_epochs = 30

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

logits = tf.matmul(X, w) + b 

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

with tf.Session() as sess:
	print sess.run(tf.shape(entropy))
	print sess.run(tf.shape(loss))


