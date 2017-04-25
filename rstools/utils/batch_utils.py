import numpy as np


def iterate_minibatches(inputs, batch_size, shuffle=True):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def generate_minibatches(inputs, batch_size, shuffle=True):
    while True:
        for data in iterate_minibatches(inputs, batch_size, shuffle):
            yield data
