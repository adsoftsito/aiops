#   Python 3.9.6
#   tensorflow==2.16.2

from pathlib import Path
import numpy as np

NUM_PUNTOS = 300
SCRIPT_DIR = Path(__file__).resolve().parent
CITY_1_PATH = SCRIPT_DIR / "city1.csv"
CITY_2_PATH = SCRIPT_DIR / "city2.csv"


def load_gps(path):
    gps = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if gps.shape != (NUM_PUNTOS, 2):
        raise ValueError(
            f"{path.name} debe contener {NUM_PUNTOS} filas con latitude,longitude"
        )
    return gps


def main():
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)

    datos_1 = load_gps(CITY_1_PATH)
    datos_2 = load_gps(CITY_2_PATH)

    X = np.concatenate([datos_1, datos_2], axis=0).astype(np.float32)

    # clase 0 = datos_1; clase 1 = datos_2
    Y = np.array(
        [0] * NUM_PUNTOS + [1] * NUM_PUNTOS,
        dtype=np.float32,
    ).reshape(-1, 1)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,), dtype=tf.float32, name="coordinates"),
            tf.keras.layers.Dense(4, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(8, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="gps_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)
        ],
    )

    model.summary()

    history = model.fit(
        X,
        Y,
        epochs=300,
        batch_size=32,
        shuffle=True,
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
    print(probabilidades[:NUM_PUNTOS][:10])

    print("\n=== Predicciones clase 1 (primeros 10) ===")
    print(probabilidades[NUM_PUNTOS:NUM_PUNTOS * 2][:10])

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            print(f"\nCapa: {layer.name}")
            print("W:")
            print(weights)
            print("b:")
            print(bias)

    model.save("gps_classifier.keras")

    export_path = "./gps-classifier/1"
    model.export(export_path)

if __name__ == "__main__":
    main()
