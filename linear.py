# adsoft 
import numpy as np
import os
#import matplotlib.pyplot as plt

# TensorFlow
import tensorflow as tf
 
print(tf.__version__)

X = np.arange(-10.0, 10.0, 1e-2)
np.random.shuffle(X)
y =  10.0 * X + 0.0

train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
                                           tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
                                           ])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
print(linear_model.predict([ [0.0], [2.0], [3.1], [4.2], [5.2] ] ).tolist() )   

export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./',export_path))

