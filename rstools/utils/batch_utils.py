import numpy as np
import types
from .os_utils import masked_if_need


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
    if not isinstance(inputs, np.ndarray):
        inputs = np.array(inputs)
    while True:
        for data in iterate_minibatches(inputs, batch_size, shuffle=shuffle, rng=rng):
            yield data


def sequence_batch(inputs, max_sequence_length=None, time_major=False, pad=0):
    """
    Args:
        inputs:
            list of sequences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
        time_major:
            boolean flag to output time-major batch
        pad:
            special symbol for sequence padding

    Outputs:
        result_batch:
            if time_major=true:
                input sequence transformed into time-major matrix
                (shape [max_time, batch_size]) padded with pad's
            else:
                input sequence transformed into batch-major matrix
                (shape [batch_size, max_time]) padded with pad's
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    max_sequence_length = max_sequence_length or max(sequence_lengths)

    result_batch = np.ones(shape=[batch_size, max_sequence_length], dtype=np.int32) * pad

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            result_batch[i, j] = element

    if time_major:
        # [batch_size, max_time] -> [max_time, batch_size]
        result_batch = result_batch.swapaxes(0, 1)

    return result_batch, sequence_lengths


def sequence_generator(data_gen, batch_params=None):
    batch_params = batch_params or {}
    for batch_seq in data_gen:
        inputs_time_major, sequence_lengths = sequence_batch(batch_seq, **batch_params)
        yield inputs_time_major, sequence_lengths


def files_iterator(mask, open_fn, shuffle=False, rng=42):
    rng = create_random_state(rng)
    files = masked_if_need(mask)
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

    files = masked_if_need(mask)
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


def merge_generators(generators, proc_fn=None):
    while True:
        data = [next(it) for it in generators]
        if proc_fn is not None:
            data = proc_fn(data)
        yield data
