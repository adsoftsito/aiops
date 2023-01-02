import os
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#print(tf.__version__)
dataset_path = keras.utils.get_file("dataset.csv", "https://firebasestorage.googleapis.com/v0/b/nocode-app-ai.appspot.com/o/datasets%2Fmelb_data.csv\?alt\=media\&token\=b891a644-e1a1-4ed7-b6a0-7bdba57c9aac")
print(dataset_path)
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
#dataset.tail()
#dataset.isna().sum()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
#dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
#model.summary()
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
#print(example_result)


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

#history = model.fit(
#  normed_train_data, train_labels,
#  epochs=EPOCHS, validation_split = 0.2, verbose=0,
#  callbacks=[PrintDot()])

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

#print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
#test_predictions = model.predict(normed_test_data).flatten()
test_predictions = model.predict([ 
    [[1.483887],	[1.865988],	[2.234620],	[1.018782],	[-2.530891],	[-1.604642],	[0.774676],	[-0.465148],	[-0.495225]],
    [[1.483887],	[1.865988],	[2.234620],	[1.018782],	[-2.530891],	[-1.604642],	[0.774676],	[-0.465148],	[-0.495225]],
    [[1.483887],	[1.865988],	[2.234620],	[1.018782],	[-2.530891],	[-1.604642],	[0.774676],	[-0.465148],	[-0.495225]]
    ])

#print(normed_test_data)
#normed_test_data.head()
print(test_predictions)
#print(test_labels)
export_path = 'linear-model/1/'
tf.saved_model.save(model, os.path.join('./',export_path))
