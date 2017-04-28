import numpy as np
import types
from .os_utils import masked_files


def create_random_state(rng):
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    return rng


def iterate_minibatches(inputs, batch_size, shuffle=False, rng=42):
    rng = create_random_state(rng)
    if shuffle:
        indices = np.arange(len(inputs))
        rng.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def generate_minibatches(inputs, batch_size, shuffle=False, rng=42):
    rng = create_random_state(rng)
    while True:
        for data in iterate_minibatches(inputs, batch_size, shuffle=shuffle, rng=rng):
            yield data


def files_iterator(mask, open_fn, shuffle=False, rng=42):
    rng = create_random_state(rng)
    files = masked_files(mask)
    for file in iterate_minibatches(files, 1, shuffle, rng=rng):
        file_data = open_fn(file[0])
        yield file_data


def files_generator(mask, open_fn, shuffle=False, rng=42):
    rng = create_random_state(rng)
    while True:
        for data in files_iterator(mask, open_fn, shuffle=shuffle, rng=rng):
            yield data


def files_data_iterator(mask, open_fn, batch_size, files_shuffle=False, data_shuffle=False, rng=42):
    rng = create_random_state(rng)

    files = masked_files(mask)
    for file in iterate_minibatches(files, 1, files_shuffle, rng=rng):
        file_data = open_fn(file[0])
        if isinstance(file_data, types.GeneratorType):
            for data in file_data:
                yield data
        else:
            for data in iterate_minibatches(file_data, batch_size, data_shuffle, rng=rng):
                yield data


def files_data_generator(
        mask, open_fn, batch_size, files_shuffle=False, data_shuffle=False, rng=42):
    rng = create_random_state(rng)
    while True:
        for data in files_data_iterator(
                mask, open_fn, batch_size,
                files_shuffle=files_shuffle, data_shuffle=data_shuffle, rng=rng):
            yield data
