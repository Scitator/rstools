import numpy as np
import itertools
import collections
from tqdm import trange
import os
import tensorflow as tf
from datetime import datetime
from .logger import Logger
from ..utils.os_utils import create_if_need, save_history, save_model
from ..visualization.plotter import plot_all_metrics


def run_generator(
        i_step, sess, run_keys, result_keys, feed_keys, data_gen,
        n_batch=-1, logger=None):
    history = collections.defaultdict(list)

    for i_batch, feed_values in enumerate(data_gen):
        run_result = sess.run(
            run_keys,
            feed_dict=dict(zip(feed_keys, feed_values)))

        for i, key in enumerate(result_keys):
            history[key].append(run_result[i])

        if logger is not None:
            for key, value in history.items():
                logger.scalar_summary(key, value, i_step + i_batch)

        if i_batch + 1 >= n_batch > 0:
            break
    return history


def run_train(sess, train_gen, train_params, val_gen=None, val_params=None, run_params=None):
    run_params = run_params or {}

    n_epochs = run_params.get("n_epochs", 100)
    log_dir = run_params.get("log_dir", "./logs_{:%Y%m%d_%H%M%S}".format(datetime.now()))
    plotter_dir = run_params.get("plotter_dir", "plotter")
    model_dir = run_params.get("model_dir", "model")
    checkpoint_every = run_params.get("checkpoint_every", 10)
    model_global_step = run_params.get("model_global_step", None)
    create_if_need(log_dir)

    logger = Logger(log_dir) if run_params.get("use_tensorboard", False) else None
    train_iter_epoch = train_params.get("n_batch", -1) < 0
    val_iter_epoch = val_params.get("n_batch", -1) < 0

    history = collections.defaultdict(list)
    saver = tf.train.Saver()

    tr = trange(
        n_epochs,
        desc="",
        leave=True)
    i_step = 0

    for i_epoch in tr:
        if train_iter_epoch:
            train_gen, train_gen_copy = itertools.tee(train_gen, 2)
        else:
            train_gen_copy = train_gen
        train_epoch_history = run_generator(
            i_step, sess, data_gen=train_gen_copy, logger=logger, **train_params)
        for metric in train_epoch_history:
            history[metric].append(np.mean(train_epoch_history[metric]))

        if val_gen is not None and val_params is not None:
            if val_iter_epoch:
                val_gen, val_gen_copy = itertools.tee(val_gen, 2)
            else:
                val_gen_copy = val_gen
            val_epoch_history = run_generator(
                i_step, sess, data_gen=val_gen_copy, logger=logger, **val_params)
            for metric in val_epoch_history:
                history[metric].append(np.mean(val_epoch_history[metric]))

        i_step += [len(value) for key, value in train_epoch_history.items()][0]  # a bit of hack
        i_epoch += 1
        if i_epoch % checkpoint_every == 0:
            checkpoint_dir = os.path.join(log_dir, model_dir, str(i_epoch))
            save_model(
                sess, saver,
                save_dir=checkpoint_dir,
                model_global_step=model_global_step)
            save_history(history, checkpoint_dir)
            plotter_checkpoint_dir = os.path.join(checkpoint_dir, plotter_dir)
            plot_all_metrics(history, save_dir=plotter_checkpoint_dir)

        desc = "\t".join(
            ["{} = {:.3f}".format(key, value[-1]) for key, value in history.items()])
        tr.set_description(desc)

    model_dir = os.path.join(log_dir, model_dir)
    save_model(
        sess, saver,
        save_dir=model_dir,
        model_global_step=model_global_step)

    save_history(history, log_dir)
    plotter_dir = os.path.join(log_dir, plotter_dir)
    plot_all_metrics(history, save_dir=plotter_dir)

    return history
