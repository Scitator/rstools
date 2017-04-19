import numpy as np
import itertools
import collections
from tqdm import trange
from .os_utils import create_if_need, save_history
from .plotter import plot_all_metrics


def run_generator(sess, run_keys, result_keys, feed_keys, data_gen, n_batch=np.inf):
    history = collections.defaultdict(list)

    for i_batch, feed_values in enumerate(data_gen):
        run_result = sess.run(
            run_keys,
            feed_dict=dict(zip(feed_keys, feed_values)))

        for i, key in enumerate(result_keys):
            history[key].append(run_result[i])

        if i_batch+1 >= n_batch:
            break
    return history


def run_train(sess, train_gen, train_params, val_gen=None, val_params=None, run_params=None):
    run_params = run_params or {}

    n_epochs = run_params.get("n_epochs", 100)
    log_dir = run_params.get("log_dir", "./logs")
    plotter_dir = run_params.get("plotter_dir", "./logs/plotter")
    create_if_need(log_dir)

    history = collections.defaultdict(list)

    tr = trange(
        n_epochs,
        desc="",
        leave=True)

    for i_epoch in tr:
        train_gen, train_gen_copy = itertools.tee(train_gen, 2)
        train_epoch_history = run_generator(sess, data_gen=train_gen, **train_params)
        for metric in train_epoch_history:
            history[metric].append(np.mean(train_epoch_history[metric]))

        if val_gen is not None and val_params is not None:
            val_gen, val_gen_copy = itertools.tee(val_gen, 2)
            val_epoch_history = run_generator(sess, data_gen=val_gen_copy, **val_params)
            for metric in val_epoch_history:
                history[metric].append(np.mean(val_epoch_history[metric]))

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[-1]) for key, value in history.items()])
        tr.set_description(desc)

    save_history(history, log_dir)
    plot_all_metrics(history, save_dir=plotter_dir)

    return history
