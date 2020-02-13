import numpy as np
import matplotlib.pyplot as plt

from libft.activations import ReLU, Softmax
from libft.layers import Dense, Input
from libft.losses import BinaryCrossentropy
from libft.metrics import BinaryAccuracy
from libft.models import Sequential
from libft.optimizers import SGD
from libft.preprocessing import to_categorical

np.seterr(all='raise')

X = np.random.rand(500, 30)
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

y = to_categorical(np.random.randint(0, 2, 500))

model = Sequential([
    Input(input_shape=(30,)),
    Dense(16),
    ReLU(),
    Dense(2),
    Softmax(),
])

model.summary()

model.compile(loss=BinaryCrossentropy(),
              optimizer=SGD(learning_rate=1e-4),
              metrics=BinaryAccuracy())

epochs = 100
history = model.fit(X_std, y,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, model.losses['training'], c='blue')
plt.plot(epoch_range, model.losses['validation'], c='red', ls='dashed')
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.show()

plt.plot(epoch_range, model.metrics['training'], c='blue')
plt.plot(epoch_range, model.metrics['validation'], c='red', ls='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
