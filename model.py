import numpy as np

def train(
    self, x_train: np.array, y_train: np.array,
    epochs: int, batch_size: int = 64,
) -> None:
    for epoch in range(epochs):
        for x_batch, y_batch in generate_batches(x_train, y_train, batch_size):
            y_hat_batch = self.__forward(x_batch)
            activation = y_hat_batch - y_batch
            self._backward(activation)
            self._update()

def predict(self, x: np.array) -> np.array:
    return self._forward(x)

