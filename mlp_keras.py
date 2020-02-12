import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    names = [
        'Id',
        'Diagnosis',
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness',
        'mean compactness',
        'mean concavity',
        'mean concave points',
        'mean symmetry',
        'mean fractal dimension',
        'se radius',
        'se texture',
        'se perimeter',
        'se area',
        'se smoothness',
        'se compactness',
        'se concavity',
        'se concave points',
        'se symmetry',
        'se fractal dimension',
        'worst radius',
        'worst texture',
        'worst perimeter',
        'worst area',
        'worst smoothness',
        'worst compactness',
        'worst concavity',
        'worst concave points',
        'worst symmetry',
        'worst fractal dimension',
    ]
    df = pd.read_csv('./data/data.csv', header=None, names=names)
    df['Diagnosis'] = (df['Diagnosis'] == 'M').astype(int)

    keeps = [
        'Diagnosis',
        'mean radius',
        'mean texture',
        'mean concavity',
        'mean fractal dimension',
        'se radius',
        'se compactness',
        'se concavity',
        'se concave points',
        'se fractal dimension',
        'worst smoothness',
        'worst concavity',
        'worst symmetry',
        'worst fractal dimension',
    ]
    df = df[keeps]

    X = df.values[:, 1:]
    y = tf.keras.utils.to_categorical(df.values[:, 0])

    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    X_std = np.copy(X)

    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_std.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    epochs = 200
    model.fit(X_std, y, epochs=epochs, validation_split=0.2)
