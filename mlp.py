import argparse
import pandas as pd

import libft.backend.math as M
from libft.activations import ReLU, Softmax
from libft.layers import Dense, Input
from libft.models import Sequential, load_model, save_model
from libft.optimizers import RMSprop
from libft.preprocessing import StandardScaler, shuffle_data, to_categorical
from libft.visualizers import plot_model


def predict(X, y, path):
    model, scaler = load_model(path)

    X = scaler.transform(X)
    loss, metric = model.evaluate(X, y, batch_size=1, verbose=1)

    size = X.shape[0]
    accuracy = M.round(metric, decimals=3)
    nb_correct = int(size * accuracy)
    print(f"> Correctly predicted: {accuracy * 100:.3}% ({nb_correct}/{size})")
    print(f"> Loss: {loss:.4}")


def training(X, y, path, verbose=False):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X, y = shuffle_data(X, y)

    shape = (X.shape[1],)
    model = Sequential([
        Input(input_shape=shape),
        Dense(16, kernel_initializer='glorot_normal', bias_initializer='ones'),
        ReLU(alpha=0.01),
        Dense(2, kernel_initializer='he_normal'),
        Softmax(),
    ])

    optimizer = RMSprop(learning_rate=1e-4)
    model.compile(optimizer, loss='bce', metrics='binary_accuracy')

    model.fit(X, y, epochs=100, validation_split=0.2, batch_size=16)
    save_model(model, scaler=scaler, path=path)
    if verbose:
        plot_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default=None,
                        required=True, help="Path to dataset.")
    parser.add_argument("-p", "--predict", type=str, default=None,
                        help="Path to saved model (Enable predict mode).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable verbose mode.")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Initialize the random number generator.")
    parser.add_argument("-o", "--output", type=str, default='saved_model.ft',
                        help="Write output to file.")

    args = parser.parse_args()

    seed = args.seed if args.seed else M.randint(0, 1e6)  # 438260
    M.random_seed(seed)

    df = pd.read_csv(args.dataset, header=None)
    df[1] = (df[1] == 'M').astype(int)
    df = df[[1, 2, 3, 8, 11, 12, 17, 18, 19, 21, 26, 28, 30, 31]]

    X = df.values[:, 1:]
    y = to_categorical(df.values[:, 0])

    if args.predict:
        predict(X, y, args.predict)
    else:
        training(X, y, args.output, verbose=args.verbose)
        print(f"seed: {seed}")
