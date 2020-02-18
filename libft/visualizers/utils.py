import libft.backend.plot as plt


def plot_model(model):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    X = range(1, len(model.losses['training']) + 1)

    loss, val_loss = model.losses['training'], model.losses['validation']
    ax[0].plot(X, loss, c='blue', label='Training')
    ax[0].plot(X, val_loss, c='red', ls='dashed', label='Validation')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    metric, val_metric = model.metrics['training'], model.metrics['validation']
    ax[1].plot(X, metric, c='blue', label='Training')
    ax[1].plot(X, val_metric, c='red', ls='dashed', label='Validation')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].legend()

    plt.show()
