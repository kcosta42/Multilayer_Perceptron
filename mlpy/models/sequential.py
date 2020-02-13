import mlpy.backend.math as M
from mlpy.layers import Input
from mlpy.layers.layer import Layer
import mlpy.losses as losses_module
import mlpy.metrics as metrics_module
import mlpy.optimizers as optimizers_module
from mlpy.preprocessing import batch_iterator, shuffle_data, train_test_split
from mlpy.visualizers import Progbar


class Sequential(object):
    """Sequential Model.

    Args:
        layers (list, optional): List of Layer objects.

    Attributes:
        losses (list): List of loss values computed during `fit`.
    """
    def __init__(self, layers=None):
        self._layers = None
        self.built = False
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer (Layer): A Layer instance.

        Raises:
            TypeError: If the `layer` argument is not a Layer instance.
        """
        if not isinstance(layer, Layer):
            raise TypeError("The added layer must be an instance of "
                            f"class Layer. Found: {layer}")
        self.built = False

        if not self._layers:
            if not isinstance(layer, Input):
                raise ValueError("Currently we need the first layer in "
                                 "a Sequential model to be an Input layer.")
            self._layers = []
            self.inputs = layer
            self.outputs = layer

        self._layers.append(layer)
        self.build()
        return self

    def build(self):
        if len(self.layers) >= 1:
            self.outputs = self.layers[-1](self.outputs)
            self.built = True

    def compile(self, optimizer, loss, metrics=None):
        """Configures the model for training.

        Args:
            optimizer (string or Optimizer): Optimizer to use.
            loss (string or Loss): Objective function.
            metrics (list): List of metrics to be evaluated by the model during
                training and testing. Example: `metrics=['accuracy']`.
        """
        self.optimizer = optimizers_module.get(optimizer)
        self.loss = losses_module.get(loss)
        self.compile_metrics = metrics_module.get(metrics)

        self.losses = {'training': [], 'validation': []}
        self.metrics = {'training': [], 'validation': []}
        # TODO We need to check if Dense followed by Activation
        # If no Activation found after Dense, we add LinearActivation layer
        for layer in self.layers:
            if hasattr(layer, 'optimize'):
                layer.optimize(self.optimizer)
        return self

    def fit(self,
            X,
            y,
            batch_size=32,
            epochs=1,
            verbose=1,
            validation_split=0.0,
            validation_data=None,
            shuffle=True):
        """Trains the model for a fixed number of epochs.

        Args:
            X (tensor or array-like): Input data.
            y (tensor or array-like): Target data.
            batch_size (integer, optional): Number of sample in gradient update.
            epochs (integer, optional): Number of epochs to train the model.
            verbose (integer, optional): Verbosity mode.
                0 = silent, 1 = one line per epoch.
            validation_split (float, optional): Fraction of the training data to
                be used as validation data. The validation data is selected from
                the last samples in the `x` and `y` data, before shuffling.
            validation_data (tuple): `(x_val, y_val)`, Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                `validation_data` will override `validation_split`.
            shuffle (boolean): Whether to shuffle the data before each epoch.
        """
        if validation_data is None:
            split = train_test_split(X, y, validation_split, shuffle=False)
            X, X_val, y, y_val = split
        else:
            X_val, y_val = validation_data

        progbar = Progbar(epochs, X.shape[0], batch_size)
        for epoch in range(epochs):
            X, y = shuffle_data(X, y)
            losses = []
            metrics = []
            for output, target in batch_iterator(X, y, batch_size):
                loss, metric = self.train_on_batch(output, target)
                losses.append(loss)
                metrics.append(metric)
                if verbose >= 1:
                    progbar.update(epoch, loss, metric)

            loss = M.mean(losses)
            metric = M.mean(metrics)
            val_loss, val_metric = self.evaluate(X_val, y_val, batch_size)
            self.losses['training'].append(loss)
            self.metrics['training'].append(metric)
            self.losses['validation'].append(val_loss)
            self.metrics['validation'].append(val_metric)
            if verbose >= 1:
                progbar.update(epoch, loss, metric, val_loss, val_metric)

    def evaluate(self, X=None, y=None, batch_size=32):
        """Returns the loss value & metrics values for the model in test mode.

        Args:
            X (tensor or array-like): Input data.
            y (tensor or array-like): Target data.
            batch_size (integer, optional): Number of sample in gradient update.
            epochs (integer, optional): Number of epochs to train the model.

        Returns:
            A scalar test loss or list of scalars with metrics.
        """
        losses = []
        metrics = []
        for output, target in batch_iterator(X, y, batch_size):
            loss, metric = self.test_on_batch(output, target)
            losses.append(loss)
            metrics.append(metric)

        return M.mean(losses), M.mean(metrics)

    def predict(self, X):
        """Generates output predictions for the input samples.

        Args:
            X (tensor or array-like): Input data.

        Returns:
            A tensor corresponding to the prediction.
        """
        output = X[:]
        for layer in self.layers:
            output = layer(output)
        return output

    def train_on_batch(self, X, y):
        """Runs a single gradient update on a single batch of data.

        Args:
            x (tensor or array-like): Input data.
            y (tensor or array-like): Target data.

        Returns:
            A scalar test loss or list of scalars with metrics.
        """
        output = X
        for layer in self.layers:
            output = layer(output)
        loss = M.mean(self.loss(y, output))
        metric = self.compile_metrics(y, output)

        grads = self.loss.gradient(y, output)
        for layer in reversed(self.layers):
            grads = layer.backward(grads)

        return loss, metric

    def test_on_batch(self, X, y):
        """Test the model on a single batch of samples.

        Args:
            X (tensor or array-like): Input data.
            y (tensor or array-like): Target data.

        Returns:
            A scalar test loss or list of scalars with metrics.
        """
        output = X
        for layer in self.layers:
            output = layer(output)
        loss = M.mean(self.loss(y, output))
        metric = self.compile_metrics(y, output)
        return loss, metric

    @property
    def layers(self):
        if self._layers and isinstance(self._layers[0], Input):
            return self._layers[1:]
        return self._layers

    def summary(self):
        """Print a summary of the layers in this model."""
        if not self.built:
            return print("Model has not yet been built.")

        line_length = 65
        positions = [int(line_length * p) for p in (0.45, 0.85, 1.0)]
        columns = ['Layer (type)', 'Output Shape', 'Param #']

        def print_row(fields, positions):
            line = ''
            for i in range(len(fields)):
                if i > 0:
                    line = line[:-1] + ' '
                line += str(fields[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            print(line)

        print(f'Model: "{self.__class__.__name__}"')
        print('_' * line_length)
        print_row(columns, positions)
        print('=' * line_length)

        total_params = 0
        layers = self.layers
        for layer in layers:
            name = layer.__class__.__name__
            total_params += layer.size
            fields = [f'({name})', layer.shape, layer.size]
            print_row(fields, positions)

        print('=' * line_length)
        print(f'Total params: {total_params}')
        print('_' * line_length)
