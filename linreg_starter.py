import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

DATA_FILE = 'data/birth_life_2010.txt'

data, n_samples = utils.read_birth_life_data(DATA_FILE)

print data.shape
print data
print n_samples

