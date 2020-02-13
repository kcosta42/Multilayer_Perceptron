import libft.backend.math as M
from libft.losses.loss import Loss


class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss when there are only two label classes
    (assumed to be 0 and 1). For each example, there should be a single
    floating-point value per prediction.

    Arguments:
        from_logits: boolean, Default: False
            Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values.
            By default, we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
    """
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        target = y_true
        output = y_pred
        if self.from_logits:
            output = 1 / (1 + M.exp(-y_pred))

        output = M.clip(output, M.epsilon(), 1.0 - M.epsilon())
        output = -target * M.log(output) - (1.0 - target) * M.log(1.0 - output)
        return M.mean(output, axis=-1)

    def gradient(self, y_true, y_pred):
        y_pred = M.clip(y_pred, M.epsilon(), 1 - M.epsilon())
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)
