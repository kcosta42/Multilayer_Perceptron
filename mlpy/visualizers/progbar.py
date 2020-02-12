import time


class Progbar(object):
    """Progress bar class.

    Args:
        epochs (integer): Total number of epochs.
        size (integer): Size of the dataset.
        batch_size (integer): Batch size.
        width (integer, optional): Width for the progress bar.
    """
    def __init__(self, epochs, size, batch_size, width=30):
        self.start = time.time()
        self.step = time.time()
        self.width = width

        self.epochs = epochs
        self.prev_epoch = -1

        self.size = size
        self.batch_size = batch_size
        self.seen_so_far = 0

        self.finished = False

    def update(self, epoch, *args):
        """Update Progress bar.

        Args:
            values (tuple): Tuple corresponding to
                (epoch, batch_size, losses, metrics)
        """

        self.step = time.time()
        self.seen_so_far = min(self.seen_so_far + self.batch_size, self.size)

        if len(args) == 2:
            val_loss, val_metric = None, None
            loss, metric = args
        else:
            loss, metric, val_loss, val_metric = args

        if epoch > self.prev_epoch:
            self.prev_epoch = epoch
            self.seen_so_far = self.batch_size
            self.start = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

        print(f"\r{self.seen_so_far}/{self.size}", end=" ")

        current = int((self.seen_so_far * self.width) / self.size)
        print(f"[{'=' * current}{'.' * (self.width - current)}]", end=" - ")

        elapsed = self.step - self.start
        print(f"{int(elapsed)}s", end=" - ")
        # print(f"{int((elapsed * 1e6) / self.seen_so_far)}us/sample", end="")

        print(f"loss: {loss:.5}", end=" - ")
        print(f"metrics: {metric:.5}", end=" - ")
        if val_loss is not None:
            print(f"val_loss: {val_loss:.5}", end=" - ")
        if val_metric is not None:
            print(f"val_metrics: {val_metric:.5}", end=" - ")

        if self.finished:
            print()

        if (epoch + 1) == self.epochs and self.seen_so_far == self.size:
            self.finished = True
