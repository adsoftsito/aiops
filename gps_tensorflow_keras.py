#   Python 3.9.6
#   tensorflow==2.16.2

import math

import numpy as np


SEED = 42


def circulo(num_datos=100, R=1, minimo=0, maximo=1):
    if num_datos <= 0:
        raise ValueError("num_datos debe ser mayor que 0")
    if minimo < 0 or maximo > 1 or minimo >= maximo:
        raise ValueError("los limites deben cumplir 0 <= minimo < maximo <= 1")

    radio_normalizado = np.sqrt(
        np.random.uniform(minimo, maximo, size=num_datos)
    )
    angulo = np.random.uniform(0, 2 * math.pi, size=num_datos)
    radio = R * radio_normalizado

    x = np.round(np.cos(angulo) * radio, 3).reshape(num_datos, 1)
    y = np.round(np.sin(angulo) * radio, 3).reshape(num_datos, 1)

    return np.column_stack([x, y]).astype(np.float32)


def main():
    import tensorflow as tf

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)

    N = 300
    datos_1 = circulo(num_datos=N, R=2)
    datos_2 = circulo(num_datos=N, R=0.5)

    X = np.concatenate([datos_1, datos_2], axis=0).astype(np.float32)
    X = np.round(X, 3)

    # clase 0 = datos_1; clase 1 = datos_2
    Y = np.array([0] * N + [1] * N, dtype=np.float32).reshape(-1, 1)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

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

    history = model.fit(
        X,
        Y,
        epochs=700,
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

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            print(f"\nCapa: {layer.name}")
            print("W:")
            print(weights)
            print("b:")
            print(bias)

    model.save("circle_classifier.keras")

    export_path = "./circle-classifier/1"
    model.export(export_path)

    print("\nSavedModel exportado en:", export_path)


if __name__ == "__main__":
    main()

