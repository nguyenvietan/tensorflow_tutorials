import tensorflow as tf
import numpy as np
import os
os.environ['TF_MIN_LOG_CPP'] = '2' # ??
import time

# read data

# create placeholders
X = tf.placeholder()

# create variables
w = tf.get_variable(tf.float32, tf.constant(0))
b = tf.get_variable(tf.float32, tf.constant(0))



