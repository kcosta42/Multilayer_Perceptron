import matplotlib.pyplot as plt
import pandas as pd

import libft.backend.math as M
from libft.layers import Dense
from libft.models import Sequential
from libft.optimizers import SGD
from libft.preprocessing import shuffle_data, to_categorical

if __name__ == '__main__':
    names = [
        'id',
        'diagnosis',
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
    df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

    keeps = [
        'diagnosis',
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
    seed = M.randint(0, 1e6)
    M.random_seed(seed)

    X = df.values[:, 1:]
    y = to_categorical(df.values[:, 0])
    X, y = shuffle_data(X, y)

    X_std = X[:]
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

    print(seed)
