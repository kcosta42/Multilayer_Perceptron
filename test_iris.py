from mlpy.layers import Dense, Input
from mlpy.activations import Sigmoid
from mlpy.models import Sequential
from mlpy.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')


if __name__ == '__main__':
    model = Sequential([
        Input(input_shape=(2,), batch_size=8),
        Dense(1, kernel_initializer='he_uniform', bias_initializer='ones'),
        Sigmoid(),
    ])

    model.summary()

    optimizer = SGD(learning_rate=0.001)
    model.compile(optimizer, loss='bce', metrics='binary_accuracy')

    df = pd.read_csv('./data/iris.csv', header=None)
    y = df.iloc[0:100, 4].values.reshape((-1, 1))
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0, 2]].values

    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    epochs = 100
    model.fit(X_std, y, epochs=epochs, validation_split=0.3)

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

    prediction = model.predict(X_std[-1])
    print(f"Prediction: {prediction} -- Original: {y[-1]}")