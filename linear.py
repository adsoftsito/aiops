# Helper libraries
import numpy as np
import os
#import matplotlib.pyplot as plt

# TensorFlow
import tensorflow as tf
 
#print(tf.__version__)
X = np.arange(-10.0, 10.0, 1e-2)
np.random.shuffle(X)
y =  2 * X + 1

