import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mlpy.layers import Dense
from mlpy.models import Sequential
from mlpy.optimizers import SGD
from mlpy.preprocessing import shuffle_data, to_categorical

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
    y = to_categorical(df.values[:, 0])
    X, y = shuffle_data(X, y)

    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

    shape = (X_std.shape[1],)
    model = Sequential([
        Dense(32, activation='relu', input_shape=shape),
        Dense(2, activation='softmax'),
    ])

    optimizer = SGD(learning_rate=3e-4)
    model.compile(optimizer, loss='bce', metrics='binary_accuracy')

    epochs = 100
    model.fit(X_std, y, epochs=epochs, validation_split=0.2)

    plt.plot(range(1, epochs + 1), model.losses['training'], c='blue')
    plt.plot(range(1, epochs + 1), model.losses['validation'], c='red', ls='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Cost function')
    plt.show()

    plt.plot(range(1, epochs + 1), model.metrics['training'], c='blue')
    plt.plot(range(1, epochs + 1), model.metrics['validation'], c='red', ls='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
