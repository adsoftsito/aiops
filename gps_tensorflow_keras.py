#   Python 3.9.6
#   tensorflow==2.16.2

import math
import numpy as np
import tensorflow as tf
from scipy import stats

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)

def circulo(num_datos=100, R=1, minimo=0, maximo=1):
    pi = math.pi

    r = (
        R
        * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos))
        * 10
    )
    theta = (
        stats.truncnorm.rvs(minimo, maximo, size=num_datos)
        * 2
        * pi
        * 10
    )

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x, 3).reshape(num_datos, 1)
    y = np.round(y, 3).reshape(num_datos, 1)

    return np.column_stack([x, y]).astype(np.float32)


# Dataset

N = 300

datos_1 = circulo(num_datos=N, R=2)
datos_2 = circulo(num_datos=N, R=0.5)

X = np.concatenate([datos_1, datos_2], axis=0).astype(np.float32)
X = np.round(X, 3)

# clase 0 = datos_1; clase 1 = datos_2
Y = np.array([0] * N + [1] * N, dtype=np.float32).reshape(-1, 1)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Arquitectura Keras
#   Entrada: 2
#   Capa oculta 1: 4 neuronas, ReLU
#   Capa oculta 2: 8 neuronas, ReLU
#   Salida: 1 neurona, Sigmoid

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,), dtype=tf.float32, name="coordinates"),
        tf.keras.layers.Dense(4, activation="relu", name="hidden_1"),
        tf.keras.layers.Dense(8, activation="relu", name="hidden_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ],
    name="circle_classifier",
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)
    ],
)

model.summary()


EPOCHS = 700

history = model.fit(
    X,
    Y,
    epochs=EPOCHS,
    batch_size=len(X),
    shuffle=False,
    verbose=1,
)

probabilidades = model.predict(X, verbose=0)
predicciones = np.round(probabilidades)

mse_final = np.mean((predicciones - Y) ** 2)
accuracy_final = np.mean(predicciones == Y)

print("\n=== RESULTADOS FINALES ===")
print("MSE sobre predicciones redondeadas:", mse_final)
print("Accuracy:", accuracy_final)

print("\n=== Predicciones clase 0 (primeros 10) ===")
print(probabilidades[:N][:10])

print("\n=== Predicciones clase 1 (primeros 10) ===")
print(probabilidades[N:N * 2][:10])

# pesos y bias de cada capa

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights, bias = layer.get_weights()
        print(f"\nCapa: {layer.name}")
        print("W:")
        print(weights)
        print("b:")
        print(bias)

# Guardar modelo Keras y exportar para TensorFlow Serving
model.save("circle_classifier.keras")

# SavedModel para TensorFlow Serving (Keras 3).
export_path = "./circle-classifier/1"
model.export(export_path)

print("\nSavedModel exportado en:", export_path)


