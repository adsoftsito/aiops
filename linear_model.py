# adsoft 
import numpy as np
import os

# TensorFlow
import tensorflow as tf
 
# print(tf.__version__)

X = np.arange(-10.0, 10.0, 1e-2)
print(X)

np.random.shuffle(X)

y =  2.0 * X + 1.0
print(y)


train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
# tf.compat.v1.global_variables_initializer()
linear_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
   ])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

print(linear_model.predict( np.array([ [0.0], [1.0], [2.0], [3.0], [4.0] ] ) ))

import shutil
import os

export_path = "./linear-model/1"

if os.path.exists(export_path):
    shutil.rmtree(export_path)

# Exportar modelo keras 3
linear_model.export(export_path)
