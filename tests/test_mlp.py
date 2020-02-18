import matplotlib.pyplot as plt
import pandas as pd

import libft.backend.math as M
from libft.activations import ReLU, Softmax
from libft.layers import Dense, Input
from libft.models import Sequential
from libft.optimizers import RMSprop
from libft.preprocessing import StandardScaler, shuffle_data, to_categorical

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
    df = pd.read_csv('./data/wdbc_training.csv', header=None, names=names)
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

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    shape = (X_std.shape[1],)
    model = Sequential([
        Input(input_shape=shape),
        Dense(16, kernel_initializer='glorot_normal', bias_initializer='ones'),
        ReLU(alpha=0.01),
        Dense(2),
        Softmax(),
    ])

    optimizer = RMSprop(learning_rate=3e-4)
    model.compile(optimizer, loss='bce', metrics='binary_accuracy')

    epochs = 100
    model.fit(X_std, y, epochs=epochs, validation_split=0.2)

    range_epoch = range(1, epochs + 1)
    plt.plot(range_epoch, model.losses['training'], c='blue')
    plt.plot(range_epoch, model.losses['validation'], c='red', ls='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Cost function')
    plt.show()

    plt.plot(range_epoch, model.metrics['training'], c='blue')
    plt.plot(range_epoch, model.metrics['validation'], c='red', ls='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    print(seed)
