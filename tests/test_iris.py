import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from libft.layers import Dense
from libft.models import Sequential
from libft.optimizers import RMSprop

np.seterr(all='raise')

model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2, )),
])

model.summary()

optimizer = RMSprop(learning_rate=1e-3)
model.compile(optimizer, loss='bce', metrics='binary_accuracy')

df = pd.read_csv('./tests/data/iris.csv', header=None)
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
model.fit(X_std, y, batch_size=8, epochs=epochs, validation_split=0.3)

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

prediction = model.predict(X_std[-1].reshape((1, -1)))
print(f"Prediction: {prediction} -- Original: {y[-1]}")
