import time

from libft.visualizers import Progbar

epochs = 25
size = 500
batch_size = 32
sleep = 0.0

progbar = Progbar(epochs, size, batch_size)
for epoch in range(epochs):
    losses = []
    metrics = []
    for i in range(size // batch_size):
        time.sleep(sleep)
        loss, metric = (10 / (i + 1)) / (epoch + 1), 0.1 * (i + 1) * (epoch + 1)
        losses.append(loss)
        metrics.append(metric)
        progbar.update(epoch, loss, metric)

    loss, metric = 10 / (epoch + 1), 0.1 * (epoch + 1)
    val_loss, val_metric = 10 / (epoch + 1), 0.1 * (epoch + 1)
    progbar.update(epoch, loss, metric, val_loss, val_metric)
